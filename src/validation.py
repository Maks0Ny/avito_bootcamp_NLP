"""Разбиение данных и функции для локальной проверки."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


SIGNATURE_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infn_params_text",
    "search_category",
]
_MISSING = "<NA>"
_SEP = "\x1f"


def _signature_value(value: Any) -> str:
    return _MISSING if value is None or pd.isna(value) else str(value)


def make_query_signature(frame: pd.DataFrame) -> pd.Series:
    missing = sorted(set(SIGNATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Cannot build query signature; missing columns: {missing}")
    normalized = frame[SIGNATURE_COLUMNS].apply(lambda col: col.map(_signature_value))
    return normalized.agg(_SEP.join, axis=1).astype("string")


def split_by_query_signature(
    train: pd.DataFrame, validation_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signatures = make_query_signature(train)
    splitter = GroupShuffleSplit(n_splits=1, test_size=validation_size, random_state=random_state)
    train_idx, valid_idx = next(splitter.split(train, groups=signatures))
    left = train.iloc[train_idx].copy()
    right = train.iloc[valid_idx].copy()
    overlap = set(make_query_signature(left)) & set(make_query_signature(right))
    if overlap:
        raise AssertionError("Query-signature leakage detected")
    return left, right


def relevant_by_signature(frame: pd.DataFrame) -> dict[str, set[str]]:
    work = frame.assign(query_signature=make_query_signature(frame))
    return work.groupby("query_signature", sort=False)["item_id"].agg(lambda x: set(map(str, x))).to_dict()
