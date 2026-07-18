"""
Abstract base class interface for text embeddings models.
"""

from abc import ABC, abstractmethod
from typing import List

class BaseEmbeddings(ABC):
    """
    Common abstraction interface for converting text to vector embeddings.
    """
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of text strings into vector formats.
        
        Args:
            texts: List of text blocks.
            
        Returns:
            List[List[float]]: List of float vectors.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single user question/query into a vector format.
        
        Args:
            text: A query string.
            
        Returns:
            List[float]: The query vector.
        """
        pass
