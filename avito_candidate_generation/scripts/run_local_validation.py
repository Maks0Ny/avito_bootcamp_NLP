#!/usr/bin/env python
"""Measure the sparse model on a held-out set of search contexts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_CONFIG, load_config, resolve_project_path
from src.data import load_train
from src.history import build_benchmark_history_signals
from src.metrics import recall_at_k
from src.pipeline import run_sparse_baseline
from src.utils import seed_everything
from src.validation import make_query_signature, relevant_by_signature, split_by_query_signature


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--max-validation-queries", type=int, default=2452)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config["random_seed"]))
    train = load_train(resolve_project_path(args.data_dir or config["data_dir"]))
    train_part, valid_part = split_by_query_signature(
        train, float(config["validation_size"]), int(config["random_seed"])
    )

    corpus = train.drop_duplicates("item_id").copy()
    valid_labeled = valid_part.assign(query_id=make_query_signature(valid_part))
    valid_queries = valid_labeled.drop_duplicates("query_id")
    if args.max_validation_queries and len(valid_queries) > args.max_validation_queries:
        valid_queries = valid_queries.sample(
            args.max_validation_queries, random_state=int(config["random_seed"])
        )
        selected = set(valid_queries["query_id"].astype(str))
        valid_labeled = valid_labeled[valid_labeled["query_id"].astype(str).isin(selected)]

    history, microcategories = build_benchmark_history_signals(
        train_part,
        valid_queries,
        set(corpus["item_id"].astype(str)),
        max_microcats=int(config["scoped_retrieval"].get("predicted_microcategories", 3)),
    )
    predictions = run_sparse_baseline(
        valid_queries,
        corpus,
        config,
        history_candidates=history,
        predicted_microcats=microcategories,
    )
    relevant = relevant_by_signature(valid_labeled)
    score = recall_at_k(predictions, relevant, int(config["final_top_k"]))
    print(f"Recall@{config['final_top_k']}: {score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
