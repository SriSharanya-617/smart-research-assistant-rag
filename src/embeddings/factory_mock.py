"""
Mock embedding provider for testing and quick dry-run environments.
"""

from typing import Dict, Any, List
from src.embeddings.base import BaseEmbeddingProvider
from src.logger import setup_logger

logger = setup_logger("mock_embeddings")

class MockEmbeddings(BaseEmbeddingProvider):
    """
    Mock Embeddings provider conformed to BaseEmbeddingProvider interface.
    """
    def __init__(self, model_name: str = "mock-model", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension
        self._embedded_count = 0
        logger.info(f"MockEmbeddings initialized simulating dimension={self.dimension}")

    def embed_documents(self, texts: List[str], normalize_embeddings: bool = True) -> List[List[float]]:
        logger.debug(f"Mock embedding {len(texts)} documents.")
        self._embedded_count += len(texts)
        # return dummy float vectors of size self.dimension
        return [[0.1] * self.dimension for _ in texts]

    def embed_query(self, text: str, normalize_embeddings: bool = True) -> List[float]:
        self._embedded_count += 1
        return [0.1] * self.dimension

    def get_dimension(self) -> int:
        return self.dimension

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "embedding_dimension": self.dimension,
            "number_of_embedded_documents": self._embedded_count,
            "processing_time": 0.0,
            "batch_size_used": 32,
            "model_loading_time": 0.0,
            "avg_embedding_time_per_document": 0.0,
            "device_used": "cpu",
            "model_name": self.model_name
        }
