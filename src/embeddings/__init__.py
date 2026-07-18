"""
Embedding package for the Smart Research Assistant.
Exposes abstract base classes, concrete model providers, and factory managers.
"""

from src.embeddings.base import BaseEmbeddingProvider, BaseEmbeddings
from src.embeddings.sentence_transformer import SentenceTransformerProvider
from src.embeddings.factory import EmbeddingFactory
from src.embeddings.exceptions import (
    EmbeddingError,
    ModelLoadError,
    EmbeddingGenerationError
)

__all__ = [
    "BaseEmbeddingProvider",
    "BaseEmbeddings",
    "SentenceTransformerProvider",
    "EmbeddingFactory",
    "EmbeddingError",
    "ModelLoadError",
    "EmbeddingGenerationError"
]
