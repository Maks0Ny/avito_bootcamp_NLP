"""Метрики качества."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence

import numpy as np


def unique_prefix(values: Iterable[str], k: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = str(value)
        if value not in seen:
            seen.add(value)
            result.append(value)
            if len(result) == k:
                break
    return result


def recall_at_k(
    predictions: Mapping[Hashable, Sequence[str]],
    relevant: Mapping[Hashable, Iterable[str]],
    k: int = 50,
) -> float:
    """Средний recall по первым уникальным *k* кандидатам каждого запроса."""
    if k <= 0:
        raise ValueError("k must be positive")
    scores: list[float] = []
    for query, labels_iter in relevant.items():
        labels = {str(value) for value in labels_iter}
        if not labels:
            raise ValueError(f"Query {query!r} has no relevant items")
        predicted = set(unique_prefix(predictions.get(query, ()), k))
        scores.append(len(predicted & labels) / len(labels))
    if not scores:
        raise ValueError("No labeled queries supplied")
    return float(np.mean(scores))
