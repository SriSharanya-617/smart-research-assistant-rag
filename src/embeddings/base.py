"""
Abstract base class interface for embedding providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from langchain_core.embeddings import Embeddings

class BaseEmbeddingProvider(Embeddings, ABC):
    """
    Common abstraction interface for converting text blocks to dense vector embeddings.
    """
    
    @abstractmethod
    def embed_documents(self, texts: List[str], normalize_embeddings: bool = True) -> List[List[float]]:
        """
        Embed a list of text strings into vector formats.
        
        Args:
            texts: List of text blocks.
            normalize_embeddings: Whether to normalize vectors to unit length.
            
        Returns:
            List[List[float]]: List of float vectors.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str, normalize_embeddings: bool = True) -> List[float]:
        """
        Embed a single user question/query into a vector format.
        
        Args:
            text: A query string.
            normalize_embeddings: Whether to normalize vector to unit length.
            
        Returns:
            List[float]: The query vector.
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Returns the output dimension of the embedding vectors.
        
        Returns:
            int: Vector space dimensions.
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Returns dictionary containing usage and performance statistics.
        
        Returns:
            Dict[str, Any]: Map of statistics metrics.
        """
        pass


# Backwards-compatible alias
BaseEmbeddings = BaseEmbeddingProvider
