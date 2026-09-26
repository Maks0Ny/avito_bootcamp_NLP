"""End-to-end sparse baseline shared by validation and benchmark inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .retrieval.base import RetrievalResult
from .retrieval.fusion import quota_union, reciprocal_rank_fusion
from .retrieval.metadata import rerank_with_metadata
from .retrieval.tfidf_char import CharTfidfRetriever
from .retrieval.tfidf_word import WordTfidfRetriever
from .text import make_item_texts, make_query_texts


def _text_options(config: Mapping[str, Any]) -> tuple[dict[str, int], bool]:
    section = config.get("text", {})
    weights = {
        "title_weight": int(section.get("title_weight", 2)),
        "params_weight": int(section.get("params_weight", 1)),
        "description_weight": int(section.get("description_weight", 1)),
    }
    return weights, bool(section.get("include_search_category", False))


def _apply_metadata(
    rankings: list[list[RetrievalResult]], queries: pd.DataFrame, items: pd.DataFrame,
    config: Mapping[str, Any],
) -> list[list[RetrievalResult]]:
    section = config.get("metadata", {})
    if not section.get("enabled", True):
        return rankings
    # Не копируем тяжёлые текстовые колонки в словарь метаданных.
    metadata_columns = ["item_id", "item_category_id", "item_location_id"]
    items_by_id = {
        str(row["item_id"]): row for row in items[metadata_columns].to_dict("records")
    }
    kwargs = {
        "category_boost": float(section.get("category_boost", 0.05)),
        "location_boost": float(section.get("location_boost", 0.02)),
        "delivery_location_boost": float(section.get("delivery_location_boost", 0.0)),
    }
    return [
        rerank_with_metadata(ranking, query, items_by_id, **kwargs)
        for ranking, query in zip(rankings, queries.to_dict("records"))
    ]


def run_sparse_baseline(
    queries: pd.DataFrame, items: pd.DataFrame, config: Mapping[str, Any],
    cache_dir: str | None = None,
    history_candidates: Mapping[str, Sequence[str]] | None = None,
    predicted_microcats: Mapping[str, Sequence[str]] | None = None,
    quota_variants: Mapping[str, Mapping[str, int]] | None = None,
    variant_predictions: dict[str, dict[str, list[str]]] | None = None,
) -> dict[str, list[str]]:
    """Retrieve candidates with independent word/char indexes and fuse rankings."""
    del cache_dir
    weights, include_category = _text_options(config)
    item_texts = make_item_texts(items, **weights)
    query_texts = make_query_texts(queries, include_category=include_category)
    item_ids = items["item_id"].astype(str).tolist()
    batch_size = int(config.get("batch_size", 64))
    retrieve_k = int(config["fusion"].get("retrieval_top_k", 100))

    scoped_config = config.get("scoped_retrieval", {})
    scoped_enabled = bool(scoped_config.get("enabled", False))
    query_ids = queries["query_id"].astype(str).tolist()
    scopes: dict[str, list[Any]] = {"global": [None] * len(queries)}
    if scoped_enabled:
        category_groups = {
            str(key): group.index.to_numpy(dtype="int64")
            for key, group in items.reset_index(drop=True).groupby("item_category_id", sort=False)
        }
        location_groups = {
            str(key): group.index.to_numpy(dtype="int64")
            for key, group in items.reset_index(drop=True).groupby("item_location_id", sort=False)
        }
        cat_location_groups = {
            (str(cat), str(loc)): group.index.to_numpy(dtype="int64")
            for (cat, loc), group in items.reset_index(drop=True).groupby(
                ["item_category_id", "item_location_id"], sort=False
            )
        }
        microcat_groups = {
            str(key): group.index.to_numpy(dtype="int64")
            for key, group in items.reset_index(drop=True).groupby("item_microcat_id", sort=False)
        }
        microloc_groups = {
            (str(micro), str(location)): group.index.to_numpy(dtype="int64")
            for (micro, location), group in items.reset_index(drop=True).groupby(
                ["item_microcat_id", "item_location_id"], sort=False
            )
        }
        predicted_count = int(scoped_config.get("predicted_microcategories", 5))
        ranked_micro_scopes: dict[str, list[Any]] = {
            **{f"micro_local_{rank}": [] for rank in range(predicted_count)},
            **{f"micro_global_{rank}": [] for rank in range(predicted_count)},
        }
        local_scopes: list[Any] = []
        micro_scopes: list[Any] = []
        for query_id, query in zip(query_ids, queries.to_dict("records")):
            category = str(query.get("search_category"))
            location = str(query.get("search_location_id"))
            local_scopes.append(
                cat_location_groups.get((category, location), location_groups.get(location, np.array([], dtype="int64")))
            )
            predicted = list((predicted_microcats or {}).get(query_id, ()))
            for rank in range(predicted_count):
                value = str(predicted[rank]) if rank < len(predicted) else ""
                ranked_micro_scopes[f"micro_local_{rank}"].append(
                    microloc_groups.get((value, location), np.array([], dtype="int64"))
                )
                ranked_micro_scopes[f"micro_global_{rank}"].append(
                    microcat_groups.get(value, np.array([], dtype="int64"))
                )
            arrays = [microcat_groups[value] for value in predicted if value in microcat_groups]
            if arrays:
                merged = np.unique(np.concatenate(arrays))
                category_indices = category_groups.get(category)
                if category_indices is not None:
                    merged = np.intersect1d(merged, category_indices, assume_unique=True)
                micro_scopes.append(merged)
            else:
                micro_scopes.append(np.array([], dtype="int64"))
        scopes["location"] = local_scopes
        category_scope_enabled = bool(
            scoped_config.get("category_scope_enabled", int(scoped_config.get("category_quota", 0)) > 0)
        )
        if category_scope_enabled:
            scopes["category"] = [
                category_groups.get(str(query.get("search_category")), np.array([], dtype="int64"))
                for query in queries.to_dict("records")
            ]
        scopes["microcategory"] = micro_scopes
        if scoped_config.get("diversify_microcategories", False):
            scopes.update(ranked_micro_scopes)

    # Два индекса строятся по очереди, иначе не хватает памяти.
    word = WordTfidfRetriever(batch_size=batch_size, **config["word_tfidf"])
    print("Fitting word TF-IDF index...")
    word.fit(item_ids, item_texts)
    print("Retrieving word candidates...")
    word_scoped = word.retrieve_scoped(query_texts, retrieve_k, scopes)
    word_scoped = {
        name: _apply_metadata(results, queries, items, config)
        for name, results in word_scoped.items()
    }
    del word

    char = CharTfidfRetriever(batch_size=batch_size, **config["char_tfidf"])
    print("Fitting character TF-IDF index...")
    char.fit(item_ids, item_texts)
    print("Retrieving character candidates...")
    char_scoped = char.retrieve_scoped(query_texts, retrieve_k, scopes)
    char_scoped = {
        name: _apply_metadata(results, queries, items, config)
        for name, results in char_scoped.items()
    }
    del char

    final_k = int(config.get("final_top_k", 50))
    fusion = config["fusion"]
    mode = fusion.get("mode", "rrf")
    output: dict[str, list[str]] = {}
    for index, query_id in enumerate(query_ids):
        if scoped_enabled:
            fused_scopes: dict[str, list[RetrievalResult]] = {}
            for scope_name in scopes:
                ids = reciprocal_rank_fusion(
                    {"word": word_scoped[scope_name][index], "char": char_scoped[scope_name][index]},
                    top_k=retrieve_k,
                    rrf_k=int(fusion.get("rrf_k", 60)),
                )
                fused_scopes[scope_name] = [RetrievalResult(item_id, 0.0) for item_id in ids]
            historical = [
                RetrievalResult(str(item_id), 0.0)
                for item_id in (history_candidates or {}).get(query_id, ())
            ]
            if scoped_config.get("diversify_microcategories", False):
                local_quotas = list(scoped_config.get("micro_local_quotas", [12, 9, 7, 4, 3]))
                global_quotas = list(scoped_config.get("micro_global_quotas", [5, 4, 3, 2, 1]))
                rankings = {}
                quotas = {}
                for rank, (local_quota, global_quota) in enumerate(zip(local_quotas, global_quotas)):
                    local_name = f"micro_local_{rank}"
                    global_name = f"micro_global_{rank}"
                    rankings[local_name] = fused_scopes[local_name]
                    rankings[global_name] = fused_scopes[global_name]
                    quotas[local_name] = int(local_quota)
                    quotas[global_name] = int(global_quota)
            else:
                rankings = {
                    "history": historical,
                    "microcategory": fused_scopes["microcategory"],
                    "location": fused_scopes["location"],
                }
                if "category" in fused_scopes:
                    rankings["category"] = fused_scopes["category"]
                rankings["global"] = fused_scopes["global"]
                quotas = {
                    "history": int(scoped_config.get("history_quota", 10)),
                    "microcategory": int(scoped_config.get("microcategory_quota", 10)),
                    "location": int(scoped_config.get("location_quota", 20)),
                    "category": int(scoped_config.get("category_quota", 0)),
                    "global": int(scoped_config.get("global_quota", 10)),
                }
            selected = quota_union(rankings, quotas, top_k=final_k)
            if quota_variants is not None:
                if variant_predictions is None:
                    raise ValueError("variant_predictions collector is required with quota_variants")
                for variant_name, variant_quotas in quota_variants.items():
                    variant_predictions.setdefault(variant_name, {})[query_id] = quota_union(
                        rankings, variant_quotas, top_k=final_k
                    )
        else:
            rankings = {"word": word_scoped["global"][index], "char": char_scoped["global"][index]}
            if mode == "rrf":
                selected = reciprocal_rank_fusion(
                    rankings, top_k=final_k, rrf_k=int(fusion.get("rrf_k", 60))
                )
            elif mode == "quota_union":
                selected = quota_union(rankings, fusion.get("quotas", {}), top_k=final_k)
            else:
                raise ValueError(f"Unknown fusion mode: {mode}")

        # Редкие запросы добираем первыми объявлениями корпуса.
        seen = set(selected)
        for item_id in item_ids:
            if len(selected) == final_k:
                break
            if item_id not in seen:
                selected.append(item_id)
                seen.add(item_id)
        output[str(query_id)] = selected
    return output
