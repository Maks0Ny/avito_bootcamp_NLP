#!/usr/bin/env python
"""Объединение sparse-кандидатов с результатами dense-модели."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import resolve_project_path
from src.data import load_benchmark
from src.submission import build_submission, write_submission
from src.text import normalize_text


def item_texts(frame: pd.DataFrame) -> list[str]:
    return [
        " ".join(filter(None, [normalize_text(title), normalize_text(params)]))
        for title, params in zip(frame["item_title_raw"], frame["item_infn_params_text"])
    ]


def query_texts(frame: pd.DataFrame) -> list[str]:
    return [
        " ".join(filter(None, [normalize_text(query), normalize_text(params)]))
        for query, params in zip(frame["search_query"], frame["search_infn_params_text"])
    ]


def dense_top_k(
    query: np.ndarray,
    item_embeddings: np.ndarray,
    indices: np.ndarray,
    item_ids: np.ndarray,
    k: int,
) -> list[str]:
    if len(indices) == 0:
        return []
    scores = item_embeddings[indices] @ query
    count = min(k, len(scores))
    selected = np.argpartition(scores, -count)[-count:]
    selected = selected[np.argsort(scores[selected])[::-1]]
    return [str(item_ids[indices[position]]) for position in selected]


def unique_fill(primary: list[str], secondary: list[str], fallback: list[str], k: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for source in (primary, secondary, fallback):
        for item_id in source:
            value = str(item_id)
            if value not in seen:
                output.append(value)
                seen.add(value)
                if len(output) == k:
                    return output
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--sparse-answer", default="answer.csv")
    parser.add_argument("--model-dir", default="artifacts/models/avito-minilm-retriever")
    parser.add_argument("--cache-dir", default="artifacts/dense_benchmark_finetuned")
    parser.add_argument("--output", default="outputs/answer_hybrid.csv")
    parser.add_argument("--predictions", default="outputs/hybrid_predictions.parquet")
    parser.add_argument("--sparse-quota", type=int, default=31)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    queries, items = load_benchmark(resolve_project_path(args.data_dir))
    item_ids = items["item_id"].astype(str).to_numpy(dtype=object)
    valid_items = set(item_ids)
    query_ids = queries["query_id"].astype(str).tolist()

    sparse_frame = pd.read_csv(resolve_project_path(args.sparse_answer), dtype=str)
    if sparse_frame["query_id"].duplicated().any():
        raise ValueError("Sparse answer contains duplicate query_id")
    sparse = {
        str(query_id): str(answer).split()
        for query_id, answer in sparse_frame[["query_id", "answer"]].itertuples(index=False, name=None)
    }

    cache = resolve_project_path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    embedding_path = cache / "item_embeddings.npy"
    id_path = cache / "item_ids.npy"
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    model = SentenceTransformer(str(resolve_project_path(args.model_dir)), device=device)
    if device.startswith("cuda"):
        model.half()
    model.max_seq_length = 128

    if embedding_path.exists() and id_path.exists():
        cached_ids = np.load(id_path, allow_pickle=True)
        if not np.array_equal(cached_ids, item_ids):
            raise ValueError("Dense benchmark cache does not match benchmark_items")
        item_embeddings = np.load(embedding_path)
    else:
        item_embeddings = model.encode(
            item_texts(items),
            batch_size=args.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")
        np.save(embedding_path, item_embeddings)
        np.save(id_path, item_ids)

    query_embeddings = model.encode(
        query_texts(queries),
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    catloc = {
        (str(category), str(location)): group.index.to_numpy(dtype="int64")
        for (category, location), group in items.reset_index(drop=True).groupby(
            ["item_category_id", "item_location_id"], sort=False
        )
    }
    fallback = item_ids.astype(str).tolist()
    predictions: dict[str, list[str]] = {}
    for position, row in enumerate(queries.to_dict("records")):
        query_id = str(row["query_id"])
        scope = catloc.get(
            (str(row["search_category"]), str(row["search_location_id"])),
            np.array([], dtype="int64"),
        )
        dense = dense_top_k(query_embeddings[position], item_embeddings, scope, item_ids, 100)
        predictions[query_id] = unique_fill(
            sparse.get(query_id, [])[: args.sparse_quota], dense, fallback, 50
        )

    prediction_path = resolve_project_path(args.predictions)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"query_id": query_ids, "item_ids": [predictions[query_id] for query_id in query_ids]}
    ).to_parquet(prediction_path, index=False)

    submission = build_submission(query_ids, predictions, valid_items, top_k=50)
    output_path = resolve_project_path(args.output)
    write_submission(submission, output_path, query_ids, valid_items, top_k=50)
    print(f"Validated hybrid submission saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
