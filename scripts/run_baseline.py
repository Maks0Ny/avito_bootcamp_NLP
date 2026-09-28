#!/usr/bin/env python
"""Run sparse hybrid retrieval for benchmark queries and save predictions parquet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_CONFIG, load_config, resolve_project_path
from src.data import load_benchmark, load_train
from src.history import build_benchmark_history_signals
from src.pipeline import run_sparse_baseline
from src.utils import ensure_directories, seed_everything


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config = load_config(args.config)
    data_dir = resolve_project_path(args.data_dir or config["data_dir"])
    output_dir = resolve_project_path(args.output_dir or config["output_dir"])
    ensure_directories(output_dir)
    seed_everything(int(config["random_seed"]))
    queries, items = load_benchmark(data_dir)
    history_candidates = None
    predicted_microcats = None
    if config.get("scoped_retrieval", {}).get("enabled", False):
        print("Building train-only historical signals...")
        train = load_train(data_dir)
        history_candidates, predicted_microcats = build_benchmark_history_signals(
            train,
            queries,
            set(items["item_id"].astype(str)),
            max_microcats=int(config["scoped_retrieval"].get("predicted_microcategories", 3)),
        )
        del train
    predictions = run_sparse_baseline(
        queries,
        items,
        config,
        history_candidates=history_candidates,
        predicted_microcats=predicted_microcats,
    )
    output = output_dir / "predictions.parquet"
    pd.DataFrame({
        "query_id": queries["query_id"].astype("string"),
        "item_ids": [predictions[str(query_id)] for query_id in queries["query_id"]],
    }).to_parquet(output, index=False)
    print(f"Saved {len(predictions)} query predictions to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
