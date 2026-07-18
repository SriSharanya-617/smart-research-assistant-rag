"""
Factory class to instantiate embedding models.
"""

from typing import List
from src.embeddings.base import BaseEmbeddings
from src.logger import setup_logger

logger = setup_logger("embedding_factory")

class LangChainEmbeddingsWrapper(BaseEmbeddings):
    """
    Wraps a LangChain embeddings class to fit our BaseEmbeddings interface.
    """
    def __init__(self, lc_embeddings):
        self.lc_embeddings = lc_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.lc_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.lc_embeddings.embed_query(text)


class MockEmbeddings(BaseEmbeddings):
    """
    A Mock Embedding model returning zero vectors.
    Useful for unit testing and local deployment validation without API keys or heavy local models.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        logger.info(f"MockEmbeddings initialized with dimension={self.dimension}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.debug(f"Mock-embedding {len(texts)} documents.")
        return [[0.1] * self.dimension for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        logger.debug(f"Mock-embedding query: '{text}'")
        return [0.1] * self.dimension


class EmbeddingFactory:
    """
    Factory to construct and return appropriate BaseEmbeddings instances.
    """
    @staticmethod
    def get_embeddings(provider: str, model_name: str, api_key: str = None) -> BaseEmbeddings:
        """
        Spawns the concrete implementation of BaseEmbeddings.
        
        Args:
            provider: 'huggingface', 'openai', or 'mock'.
            model_name: Specific model string name.
            api_key: Optional API key.
            
        Returns:
            BaseEmbeddings: Concrete embedding wrapper.
        """
        provider_clean = provider.lower().strip()
        logger.info(f"Instantiating embedding model: provider={provider_clean}, model={model_name}")

        try:
            if provider_clean == "huggingface":
                # import lazily to speed up startup
                from langchain_huggingface import HuggingFaceEmbeddings
                lc_embed = HuggingFaceEmbeddings(model_name=model_name)
                return LangChainEmbeddingsWrapper(lc_embed)

            elif provider_clean == "openai":
                from langchain_openai import OpenAIEmbeddings
                lc_embed = OpenAIEmbeddings(model=model_name, api_key=api_key)
                return LangChainEmbeddingsWrapper(lc_embed)

            elif provider_clean == "mock":
                return MockEmbeddings()

            else:
                logger.warning(f"Unknown embedding provider '{provider}'. Falling back to MockEmbeddings.")
                return MockEmbeddings()

        except Exception as e:
            logger.error(f"Failed to load embedding provider '{provider}': {e}. Falling back to MockEmbeddings.")
            return MockEmbeddings()
