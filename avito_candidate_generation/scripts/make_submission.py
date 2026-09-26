#!/usr/bin/env python
"""Convert saved predictions into a strictly validated answer.csv."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import resolve_project_path
from src.data import load_benchmark
from src.submission import build_submission, write_submission


def _as_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, str):
        return [str(item) for item in value.tolist()]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            parsed = ast.literal_eval(stripped)
            return [str(item) for item in parsed]
        return [] if not stripped else stripped.split()
    raise TypeError(f"Unsupported prediction value: {type(value).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--predictions", default="outputs/predictions.parquet")
    parser.add_argument("--output", default="answer.csv")
    args = parser.parse_args()
    queries, items = load_benchmark(resolve_project_path(args.data_dir))
    prediction_path = resolve_project_path(args.predictions)
    raw = pd.read_parquet(prediction_path)
    if "query_id" not in raw or not ({"item_ids", "answer"} & set(raw.columns)):
        raise ValueError("Predictions need query_id plus item_ids or answer column")
    value_column = "item_ids" if "item_ids" in raw else "answer"
    if raw["query_id"].astype(str).duplicated().any():
        raise ValueError("Predictions contain duplicate query_id")
    predictions = {
        str(query_id): _as_list(value)
        for query_id, value in raw[["query_id", value_column]].itertuples(index=False, name=None)
    }
    query_ids = queries["query_id"].astype(str).tolist()
    valid_items = set(items["item_id"].astype(str))
    submission = build_submission(query_ids, predictions, valid_items, top_k=50)
    output = resolve_project_path(args.output)
    write_submission(submission, output, query_ids, valid_items, top_k=50)
    print(f"Validated submission saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
