"""Символьный TF-IDF для опечаток и вариантов написания."""

from __future__ import annotations

from typing import Any

from .tfidf_word import WordTfidfRetriever


class CharTfidfRetriever(WordTfidfRetriever):
    def __init__(self, batch_size: int = 64, **vectorizer_params: Any) -> None:
        params = dict(vectorizer_params)
        params.setdefault("analyzer", "char_wb")
        params.setdefault("ngram_range", (3, 5))
        super().__init__(batch_size=batch_size, **params)
