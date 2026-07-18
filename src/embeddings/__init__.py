"""
Embeddings module containing common embedding models interfaces.
"""

from src.embeddings.base import BaseEmbeddings
from src.embeddings.factory import EmbeddingFactory

__all__ = ["BaseEmbeddings", "EmbeddingFactory"]
