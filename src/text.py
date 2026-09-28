"""Подготовка текста для поиска."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping

import pandas as pd


_SPACE_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalize_text(value: Any) -> str:
    """Нормализовать регистр, Unicode, пропуски и пробелы."""
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    return _SPACE_RE.sub(" ", text).strip()


def _repeat(value: Any, weight: int) -> list[str]:
    text = normalize_text(value)
    return [text] * max(0, int(weight)) if text else []


def build_query_text(row: Mapping[str, Any], include_category: bool = False) -> str:
    parts = [normalize_text(row.get("search_query")), normalize_text(row.get("search_infn_params_text"))]
    if include_category:
        category = normalize_text(row.get("search_category"))
        if category:
            parts.append(f"category_{category}")
    return " ".join(part for part in parts if part)


def build_item_text(
    row: Mapping[str, Any], title_weight: int = 2, params_weight: int = 1,
    description_weight: int = 1,
) -> str:
    parts: list[str] = []
    parts += _repeat(row.get("item_title_raw"), title_weight)
    parts += _repeat(row.get("item_infn_params_text"), params_weight)
    parts += _repeat(row.get("item_description_raw"), description_weight)
    return " ".join(parts)


def make_query_texts(frame: pd.DataFrame, include_category: bool = False) -> list[str]:
    return [build_query_text(row, include_category) for row in frame.to_dict("records")]


def make_item_texts(frame: pd.DataFrame, **weights: int) -> list[str]:
    return [build_item_text(row, **weights) for row in frame.to_dict("records")]
