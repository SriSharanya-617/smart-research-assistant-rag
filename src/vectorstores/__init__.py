"""
Vector Store Package for the Smart Research Assistant.
Exposes database interfaces, ChromaDB & FAISS wrappers, factories, and exception classes.
"""

from src.vectorstores.base import BaseVectorStore, validate_vector
from src.vectorstores.chroma_store import ChromaVectorStore, ChromaStore
from src.vectorstores.faiss_store import FAISSVectorStore, FAISSStore
from src.vectorstores.manager import VectorStoreFactory, VectorStoreManager
from src.vectorstores.exceptions import (
    VectorStoreError,
    IndexLoadError,
    DimensionMismatchError,
    DuplicateVectorError,
    CorruptedIndexError
)

__all__ = [
    "BaseVectorStore",
    "validate_vector",
    "ChromaVectorStore",
    "ChromaStore",
    "FAISSVectorStore",
    "FAISSStore",
    "VectorStoreFactory",
    "VectorStoreManager",
    "VectorStoreError",
    "IndexLoadError",
    "DimensionMismatchError",
    "DuplicateVectorError",
    "CorruptedIndexError"
]
