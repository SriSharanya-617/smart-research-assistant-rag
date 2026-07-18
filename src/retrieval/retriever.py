"""
Search Engine / Retriever class executing query lookups against vector indexes.
"""

from typing import List
from src.ingestion.base import Document
from src.logger import setup_logger
from src.vectorstores.base import BaseVectorStore

logger = setup_logger("retriever")

class SearchEngine:
    """
    SearchEngine acts as the retrieval coordinator.
    It takes raw queries, interacts with the active Vector Store, and applies filters.
    """
    def __init__(self, vector_store: BaseVectorStore):
        self.vector_store = vector_store
        logger.info("SearchEngine retriever interface successfully initialized.")

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Queries the vector store for top-K matching documents.
        
        Args:
            query: The user search text.
            k: Number of matches.
            
        Returns:
            List[Document]: Similar documents list.
        """
        logger.info(f"Executing retrieval query lookup: '{query}' (k={k})")
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Retrieved {len(results)} source documents from database.")
            return results
        except Exception as e:
            logger.error(f"Failed to execute search query retrieval: {e}")
            return []
