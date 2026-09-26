"""Soft metadata signals; no category or location is a hard filter."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from .base import RetrievalResult


def _equal_nonmissing(left: Any, right: Any) -> bool:
    return not pd.isna(left) and not pd.isna(right) and str(left) == str(right)


def rerank_with_metadata(
    results: Sequence[RetrievalResult], query: Mapping[str, Any], items_by_id: Mapping[str, Mapping[str, Any]],
    category_boost: float = 0.05, location_boost: float = 0.02,
    delivery_location_boost: float = 0.0,
) -> list[RetrievalResult]:
    delivery = bool(query.get("search_is_delivery_search", False))
    rescored: list[RetrievalResult] = []
    for result in results:
        item = items_by_id[result.item_id]
        score = result.score
        score += category_boost * _equal_nonmissing(query.get("search_category"), item.get("item_category_id"))
        loc_boost = delivery_location_boost if delivery else location_boost
        score += loc_boost * _equal_nonmissing(query.get("search_location_id"), item.get("item_location_id"))
        rescored.append(RetrievalResult(result.item_id, float(score)))
    return sorted(rescored, key=lambda value: value.score, reverse=True)


def metadata_flags(query: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, int]:
    return {
        "category_match": int(_equal_nonmissing(query.get("search_category"), item.get("item_category_id"))),
        "location_match": int(_equal_nonmissing(query.get("search_location_id"), item.get("item_location_id"))),
        "delivery_search": int(bool(query.get("search_is_delivery_search", False))),
    }
