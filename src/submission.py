"""Сборка и проверка answer.csv."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from .metrics import unique_prefix


QUERY_RE = re.compile(r"^.{16}$", flags=re.DOTALL)
ITEM_RE = re.compile(r"^[0-9a-f]{16}$")


def _prediction_mapping(
    query_ids: Sequence[str], predictions: Mapping[str, Sequence[str]] | Sequence[Sequence[str]]
) -> dict[str, Sequence[str]]:
    ids = [str(value) for value in query_ids]
    if isinstance(predictions, Mapping):
        return {str(key): value for key, value in predictions.items()}
    if len(ids) != len(predictions):
        raise ValueError("Prediction list length differs from query_ids length")
    return dict(zip(ids, predictions))


def build_submission(
    query_ids: Sequence[str], predictions: Mapping[str, Sequence[str]] | Sequence[Sequence[str]],
    valid_item_ids: Sequence[str] | set[str], top_k: int = 50,
) -> pd.DataFrame:
    ids = [str(value) for value in query_ids]
    mapping = _prediction_mapping(ids, predictions)
    if set(mapping) != set(ids):
        missing = sorted(set(ids) - set(mapping))[:5]
        extra = sorted(set(mapping) - set(ids))[:5]
        raise ValueError(f"Prediction query IDs mismatch; missing={missing}, extra={extra}")
    rows = [{"query_id": query_id, "answer": " ".join(unique_prefix(mapping[query_id], top_k))} for query_id in ids]
    frame = pd.DataFrame(rows, columns=["query_id", "answer"], dtype="string")
    validate_submission(frame, ids, valid_item_ids, top_k=top_k)
    return frame


def validate_submission(
    frame: pd.DataFrame, benchmark_query_ids: Sequence[str], valid_item_ids: Sequence[str] | set[str],
    top_k: int = 50,
) -> None:
    if list(frame.columns) != ["query_id", "answer"]:
        raise ValueError("Submission must have exactly two ordered columns: query_id, answer")
    if frame.isna().any().any():
        raise ValueError("Submission contains NaN")
    expected = [str(value) for value in benchmark_query_ids]
    actual = frame["query_id"].astype(str).tolist()
    if len(frame) != len(expected) or len(actual) != len(set(actual)):
        raise ValueError("Wrong row count or duplicate query_id")
    if set(actual) != set(expected):
        raise ValueError("Submission query IDs differ from benchmark query IDs")
    if not all(QUERY_RE.fullmatch(value) for value in actual):
        raise ValueError("Every query_id must be a 16-character string")

    valid_items = {str(value) for value in valid_item_ids}
    for query_id, answer in frame[["query_id", "answer"]].itertuples(index=False, name=None):
        answer = str(answer)
        if any(token in answer for token in ("[", "]", "'", '"', ",")):
            raise ValueError(f"Invalid list/comma representation for query {query_id}")
        values = [] if answer == "" else answer.split(" ")
        if "" in values:
            raise ValueError(f"Whitespace is not exactly one space for query {query_id}")
        if len(values) > top_k or len(values) != len(set(values)):
            raise ValueError(f"Too many or duplicate item IDs for query {query_id}")
        for item_id in values:
            if not ITEM_RE.fullmatch(item_id):
                raise ValueError(f"Invalid item_id {item_id!r} for query {query_id}")
            if item_id not in valid_items:
                raise ValueError(f"Unknown item_id {item_id!r} for query {query_id}")


def write_submission(
    frame: pd.DataFrame, output_path: str | Path, benchmark_query_ids: Sequence[str],
    valid_item_ids: Sequence[str] | set[str], top_k: int = 50,
) -> Path:
    validate_submission(frame, benchmark_query_ids, valid_item_ids, top_k)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8")
    reread = pd.read_csv(output, dtype={"query_id": "string", "answer": "string"}, keep_default_na=False)
    validate_submission(reread, benchmark_query_ids, valid_item_ids, top_k)
    return output
