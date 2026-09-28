"""Sparse-поиск кандидатов."""

from .base import RetrievalResult, Retriever
from .tfidf_char import CharTfidfRetriever
from .tfidf_word import WordTfidfRetriever

__all__ = ["RetrievalResult", "Retriever", "WordTfidfRetriever", "CharTfidfRetriever"]
