"""Общие типы для поиска кандидатов."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class RetrievalResult:
    item_id: str
    score: float


class Retriever(Protocol):
    def fit(self, item_ids: Sequence[str], item_texts: Sequence[str]) -> "Retriever": ...
    def retrieve(self, query_texts: Sequence[str], top_k: int) -> list[list[RetrievalResult]]: ...
