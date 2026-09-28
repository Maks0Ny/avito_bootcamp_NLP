"""Memory-bounded word TF-IDF retrieval."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

from .base import RetrievalResult


class WordTfidfRetriever:
    """Fit a sparse item index and score queries in small batches.

    Query-by-item similarity is never materialized as one global dense matrix.
    Each sparse row is top-k selected independently.
    """

    def __init__(self, batch_size: int = 64, **vectorizer_params: Any) -> None:
        params = dict(vectorizer_params)
        params.setdefault("analyzer", "word")
        params.setdefault("ngram_range", (1, 2))
        params.setdefault("dtype", np.float32)
        params["ngram_range"] = tuple(params["ngram_range"])
        self.batch_size = int(batch_size)
        self.vectorizer = TfidfVectorizer(**params)
        self.item_matrix: sparse.csr_matrix | None = None
        self.item_ids: np.ndarray | None = None

    def fit(self, item_ids: Sequence[str], item_texts: Sequence[str]) -> "WordTfidfRetriever":
        if len(item_ids) != len(item_texts):
            raise ValueError("item_ids and item_texts must have equal length")
        self.item_ids = np.asarray([str(value) for value in item_ids], dtype=object)
        self.item_matrix = self.vectorizer.fit_transform(item_texts).tocsr().astype(np.float32)
        return self

    def retrieve(self, query_texts: Sequence[str], top_k: int) -> list[list[RetrievalResult]]:
        return self.retrieve_scoped(
            query_texts, top_k, {"all": [None] * len(query_texts)}
        )["all"]

    def retrieve_scoped(
        self,
        query_texts: Sequence[str],
        top_k: int,
        scopes: Mapping[str, Sequence[np.ndarray | None]],
    ) -> dict[str, list[list[RetrievalResult]]]:
        """Retrieve several filtered rankings from a single similarity pass.

        Each scope supplies sorted item-matrix row indices for every query, or
        ``None`` for the full corpus. This makes location/microcategory candidate
        lists cheap: sparse dot products are calculated only once.
        """
        if self.item_matrix is None or self.item_ids is None:
            raise RuntimeError("Retriever is not fitted")
        top_k = min(int(top_k), len(self.item_ids))
        for name, allowed in scopes.items():
            if len(allowed) != len(query_texts):
                raise ValueError(f"Scope {name!r} length differs from query count")
        output: dict[str, list[list[RetrievalResult]]] = {name: [] for name in scopes}
        for start in tqdm(range(0, len(query_texts), self.batch_size), desc="TF-IDF batches", leave=False):
            query_matrix = self.vectorizer.transform(query_texts[start : start + self.batch_size])
            similarities = (query_matrix @ self.item_matrix.T).tocsr()
            for row_index in range(similarities.shape[0]):
                row = similarities.getrow(row_index)
                absolute_query_index = start + row_index
                for name, allowed_per_query in scopes.items():
                    allowed = allowed_per_query[absolute_query_index]
                    indices = row.indices
                    scores = row.data
                    if allowed is not None:
                        allowed = np.asarray(allowed, dtype=np.int64)
                        if len(allowed) == 0 or row.nnz == 0:
                            output[name].append([])
                            continue
                        positions = np.searchsorted(allowed, indices)
                        valid = positions < len(allowed)
                        mask = np.zeros(len(indices), dtype=bool)
                        mask[valid] = allowed[positions[valid]] == indices[valid]
                        indices = indices[mask]
                        scores = scores[mask]
                    if len(scores) == 0:
                        output[name].append([])
                        continue
                    count = min(top_k, len(scores))
                    chosen = np.argpartition(scores, -count)[-count:]
                    chosen = chosen[np.argsort(scores[chosen])[::-1]]
                    output[name].append([
                        RetrievalResult(str(self.item_ids[indices[pos]]), float(scores[pos]))
                        for pos in chosen
                    ])
        return output
