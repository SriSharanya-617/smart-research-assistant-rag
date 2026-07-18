"""
Factory class to instantiate and cache embedding providers as singletons.
"""

from typing import Dict, Tuple, Optional
from src.embeddings.base import BaseEmbeddingProvider
from src.embeddings.sentence_transformer import SentenceTransformerProvider
from src.embeddings.exceptions import EmbeddingError
from src.logger import setup_logger
from src.config import get_config

logger = setup_logger("embedding_factory")

# Global singleton cache registry
_embeddings_cache: Dict[Tuple[str, str], BaseEmbeddingProvider] = {}

class EmbeddingFactory:
    """
    Factory class acting as the single source of truth for embedding providers.
    Manages caching to prevent reloading heavy models into RAM.
    """
    
    @staticmethod
    def get_embeddings(
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        cache_folder: Optional[str] = None
    ) -> BaseEmbeddingProvider:
        """
        Retrieves or instantiates an embedding provider.
        
        Args:
            provider: 'huggingface' (or 'sentence-transformers'), or 'mock'. Defaults to config.
            model_name: Specific model ID string. Defaults to config.
            device: 'cpu', 'cuda' or None for auto.
            batch_size: batch size. Defaults to config.
            cache_folder: cache folder path. Defaults to config.
            
        Returns:
            BaseEmbeddingProvider: The cached singleton instance.
        """
        config = get_config()
        
        # Load from config defaults if parameters are omitted
        prov_clean = (provider or config.EMBEDDING_PROVIDER).lower().strip()
        model_clean = (model_name or config.EMBEDDING_MODEL).strip()
        dev_clean = device or config.EMBEDDING_DEVICE
        bs_clean = batch_size or config.EMBEDDING_BATCH_SIZE
        cache_dir_clean = cache_folder or config.EMBEDDING_CACHE_DIR

        # Create cache key
        cache_key = (prov_clean, model_clean)
        
        # Check cache registry
        if cache_key in _embeddings_cache:
            logger.info(f"Retrieving cached embedding provider singleton for key: {cache_key}")
            return _embeddings_cache[cache_key]

        logger.info(f"Creating new embedding provider instance for key: {cache_key}")
        
        try:
            if prov_clean in ["huggingface", "sentence-transformers", "sentence_transformers"]:
                provider_inst = SentenceTransformerProvider(
                    model_name=model_clean,
                    device=dev_clean,
                    batch_size=bs_clean,
                    cache_folder=cache_dir_clean
                )
            elif prov_clean == "mock":
                # Lazy import to avoid circular dependencies
                from src.embeddings.factory_mock import MockEmbeddings
                provider_inst = MockEmbeddings(model_name=model_clean)
            else:
                logger.warning(
                    f"Unsupported embedding provider '{provider}'. "
                    f"Defaulting to HuggingFace SentenceTransformerProvider."
                )
                provider_inst = SentenceTransformerProvider(
                    model_name=model_clean,
                    device=dev_clean,
                    batch_size=bs_clean,
                    cache_folder=cache_dir_clean
                )
                
            # Cache the instance
            _embeddings_cache[cache_key] = provider_inst
            return provider_inst
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding provider: {e}")
            raise EmbeddingError(f"Factory could not construct provider: {e}")

    @staticmethod
    def clear_cache() -> None:
        """
        Clears the cached embedding providers.
        """
        global _embeddings_cache
        logger.info("Clearing embedding providers cache.")
        _embeddings_cache.clear()
