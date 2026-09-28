"""Признаки из истории обучающих запросов."""

from __future__ import annotations

import pandas as pd

from .text import normalize_text
from .validation import make_query_signature


def build_benchmark_history_signals(
    train: pd.DataFrame,
    queries: pd.DataFrame,
    valid_item_ids: set[str],
    max_microcats: int = 3,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Собрать исторические объявления и вероятные микрокатегории запросов."""
    work = train.copy()
    work["_text"] = work["search_query"].map(normalize_text)
    work["_signature"] = make_query_signature(work)
    work["item_id"] = work["item_id"].astype(str)
    work = work[work["item_id"].isin(valid_item_ids)]

    def ranked_map(key: str, value: str) -> dict[str, list[str]]:
        counts = (
            work.groupby([key, value], sort=False)
            .size()
            .rename("count")
            .reset_index()
            .sort_values([key, "count", value], ascending=[True, False, True])
        )
        return counts.groupby(key, sort=False)[value].agg(lambda values: [str(v) for v in values]).to_dict()

    by_signature = ranked_map("_signature", "item_id")
    by_text = ranked_map("_text", "item_id")

    # Для микрокатегории достаточно истории запросов, наличие item в корпусе не важно.
    full = train.copy()
    full["_text"] = full["search_query"].map(normalize_text)
    micro_counts = (
        full.groupby(["_text", "item_microcat_id"], sort=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["_text", "count", "item_microcat_id"], ascending=[True, False, True])
    )
    micro_by_text = micro_counts.groupby("_text", sort=False)["item_microcat_id"].agg(
        lambda values: [str(value) for value in values[:max_microcats]]
    ).to_dict()

    query_signatures = make_query_signature(queries)
    query_texts = queries["search_query"].map(normalize_text)
    candidates: dict[str, list[str]] = {}
    microcats: dict[str, list[str]] = {}
    for query_id, signature, text in zip(queries["query_id"].astype(str), query_signatures, query_texts):
        ordered: list[str] = []
        seen: set[str] = set()
        for item_id in by_signature.get(str(signature), []) + by_text.get(text, []):
            if item_id not in seen:
                seen.add(item_id)
                ordered.append(item_id)
        candidates[query_id] = ordered
        microcats[query_id] = micro_by_text.get(text, [])
    return candidates, microcats
