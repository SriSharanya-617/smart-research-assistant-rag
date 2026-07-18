"""
Abstract base class interface for Vector Stores.
Includes unified vector validation routines.
"""

import math
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from src.ingestion.base import Document
from src.vectorstores.exceptions import DimensionMismatchError

def validate_vector(vector: List[float], expected_dim: Optional[int] = None) -> None:
    """
    Validates a vector embedding's dimension, and checks for NaNs/Infs.
    
    Args:
        vector: List of float values representing an embedding.
        expected_dim: Optional expected dimension to assert.
        
    Raises:
        ValueError: If vector is empty, contains NaN, or infinite values.
        DimensionMismatchError: If dimension does not match expected_dim.
    """
    if not vector:
        raise ValueError("Vector embedding cannot be empty.")
        
    if expected_dim is not None and len(vector) != expected_dim:
        raise DimensionMismatchError(
            f"Embedding vector dimension {len(vector)} does not match index dimension {expected_dim}."
        )
        
    for idx, val in enumerate(vector):
        if math.isnan(val):
            raise ValueError(f"Vector contains NaN value at index {idx}.")
        if math.isinf(val):
            raise ValueError(f"Vector contains infinite value ({val}) at index {idx}.")


class BaseVectorStore(ABC):
    """
    Common abstraction layer for all vector database integrations.
    Decoupled from specific DB library APIs.
    """
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """
        Embeds and indexes documents in the vector database.
        
        Args:
            documents: List of Document objects to insert.
        """
        pass

    @abstractmethod
    def update_documents(self, documents: List[Document]) -> None:
        """
        Updates existing documents in the database matching document/chunk IDs.
        
        Args:
            documents: List of updated Document objects.
        """
        pass

    @abstractmethod
    def remove_documents(self, document_ids: List[str]) -> None:
        """
        Removes documents matching the specified document or chunk IDs.
        
        Args:
            document_ids: List of unique document/chunk identifiers.
        """
        pass

    @abstractmethod
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        search_type: str = "dense",
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        Performs similarity search returning documents with similarity scores.
        
        Args:
            query: Question or search string.
            k: Top-K matches.
            filter: Metadata filter dictionary.
            search_type: "dense" (vector search), "sparse" (keyword search), or "hybrid".
            
        Returns:
            List[Tuple[Document, float]]: List of tuples containing Document and score.
        """
        pass

    @abstractmethod
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Executes Maximal Marginal Relevance (MMR) search for diverse retrieval.
        
        Args:
            query: Question or search string.
            k: Final matching count.
            fetch_k: Candidates count to fetch before reranking.
            lambda_mult: Diversity weight (0.0=diverse, 1.0=relevant).
            filter: Metadata filter.
            
        Returns:
            List[Document]: List of diverse Documents.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Persists the index to disk.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Loads the index from disk.
        """
        pass

    @abstractmethod
    def delete_index(self) -> None:
        """
        Deletes the physical index directory from disk.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Resets and clears the vector database in-memory/disk indexes.
        """
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Exposes usage and query execution metrics.
        """
        pass

    @abstractmethod
    def get_index_info(self) -> Dict[str, Any]:
        """
        Exposes static properties of the loaded index.
        """
        pass
