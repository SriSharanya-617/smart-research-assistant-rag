"""
Abstract base class interface for Retrievers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from src.ingestion.base import Document
from src.vectorstores.base import BaseVectorStore

class BaseRetriever(ABC):
    """
    Common abstraction interface for RAG chunk retrieval modules.
    Decoupled from direct vector database library APIs.
    """
    def __init__(self, vector_store: BaseVectorStore):
        """
        Args:
            vector_store: Instantiated vector store database backend.
        """
        self.vector_store = vector_store

    @abstractmethod
    def retrieve(
        self,
        query: str,
        limit: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Searches the database and returns matching document chunks with similarity scores.
        
        Args:
            query: Cleaned query query string.
            limit: Number of records to return.
            filter: Metadata filters dict.
            
        Returns:
            List[Tuple[Document, float]]: Sorted documents with distance/similarity scores.
        """
        pass
