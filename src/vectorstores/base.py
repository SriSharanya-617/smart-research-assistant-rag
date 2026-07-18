"""
Abstract base class interface for Vector Store databases.
"""

from abc import ABC, abstractmethod
from typing import List
from src.ingestion.base import Document

class BaseVectorStore(ABC):
    """
    Common abstraction interface for Vector Store operations.
    Supports index management, document insertion, and similarity queries.
    """
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """
        Embeds and adds documents to the vector database.
        
        Args:
            documents: List of Document objects to insert.
        """
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Executes a similarity search against the vector database.
        
        Args:
            query: The user question/query text.
            k: Top-K similar items to return.
            
        Returns:
            List[Document]: List of matching Document objects containing metadata.
        """
        pass

    @abstractmethod
    def delete_index(self) -> None:
        """
        Clears or deletes the current database index.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Persists the vector index to a local directory path.
        
        Args:
            path: Directory path.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Loads the vector index from a local directory path.
        
        Args:
            path: Directory path.
        """
        pass
