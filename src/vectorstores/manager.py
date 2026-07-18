"""
VectorStoreFactory class to instantiate and cache vector stores as singletons.
"""

from typing import Dict, Tuple, Optional
from src.embeddings.base import BaseEmbeddingProvider
from src.vectorstores.base import BaseVectorStore
from src.vectorstores.chroma_store import ChromaVectorStore
from src.vectorstores.faiss_store import FAISSVectorStore
from src.logger import setup_logger
from src.config import get_config

logger = setup_logger("vector_store_factory")

# Global singleton cache registry
_vector_stores_cache: Dict[Tuple[str, str], BaseVectorStore] = {}

class VectorStoreFactory:
    """
    Factory class responsible for creating and retrieving the appropriate vector database store.
    Caches loaded vector indexes to prevent repeatedly loading resources from disk.
    """
    @staticmethod
    def get_vector_store(
        store_type: Optional[str] = None,
        embeddings: Optional[BaseEmbeddingProvider] = None,
        persist_dir: Optional[str] = None
    ) -> BaseVectorStore:
        """
        Retrieves or instantiates the concrete vector store wrapper based on requested type.
        
        Args:
            store_type: 'chroma' or 'faiss'. Defaults to config.
            embeddings: Embeddings model instance. Defaults to factory mock.
            persist_dir: Path to persist files. Defaults to config.
            
        Returns:
            BaseVectorStore: The active vector database wrapper singleton.
        """
        config = get_config()
        
        # Load from config defaults if parameters are omitted
        store_clean = (store_type or config.VECTOR_STORE_TYPE).lower().strip()
        dir_clean = (persist_dir or config.VECTOR_STORE_DIR).strip()
        
        # Resolve embeddings if None
        if embeddings is None:
            from src.embeddings.factory import EmbeddingFactory
            embeddings = EmbeddingFactory.get_embeddings()

        # Cache key mapping
        cache_key = (store_clean, dir_clean)
        
        # Check cache registry
        if cache_key in _vector_stores_cache:
            logger.info(f"Retrieving cached vector store singleton for key: {cache_key}")
            return _vector_stores_cache[cache_key]

        logger.info(f"Creating new vector store instance for key: {cache_key}")
        
        if store_clean == "chroma":
            store_inst = ChromaVectorStore(embeddings=embeddings, persist_directory=dir_clean)
        elif store_clean == "faiss":
            store_inst = FAISSVectorStore(embeddings=embeddings, persist_directory=dir_clean)
        else:
            logger.warning(f"Unsupported store type '{store_type}'. Defaulting to ChromaVectorStore.")
            store_inst = ChromaVectorStore(embeddings=embeddings, persist_directory=dir_clean)
            
        # Cache instance
        _vector_stores_cache[cache_key] = store_inst
        return store_inst

    @staticmethod
    def clear_cache() -> None:
        """
        Clears the cached vector store instances.
        """
        global _vector_stores_cache
        logger.info("Clearing vector store cache.")
        _vector_stores_cache.clear()


# Backwards-compatible alias
VectorStoreManager = VectorStoreFactory
