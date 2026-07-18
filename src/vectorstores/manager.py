"""
Manager module for dynamic selection and initialization of Vector Stores.
"""

from src.embeddings.base import BaseEmbeddings
from src.logger import setup_logger
from src.vectorstores.base import BaseVectorStore
from src.vectorstores.chroma_store import ChromaStore
from src.vectorstores.faiss_store import FAISSStore

logger = setup_logger("vector_store_manager")

class VectorStoreManager:
    """
    Manager class responsible for creating and retrieving the appropriate vector database store.
    """
    @staticmethod
    def get_vector_store(
        store_type: str,
        embeddings: BaseEmbeddings,
        persist_dir: str
    ) -> BaseVectorStore:
        """
        Retrieves the concrete vector store wrapper based on requested type.
        
        Args:
            store_type: 'chroma' or 'faiss'.
            embeddings: Embeddings model instance.
            persist_dir: Path to persist files.
            
        Returns:
            BaseVectorStore: The active vector database wrapper.
        """
        store_clean = store_type.lower().strip()
        logger.info(f"Retrieving Vector Store: type={store_clean}, directory={persist_dir}")
        
        if store_clean == "chroma":
            return ChromaStore(embeddings=embeddings, persist_directory=persist_dir)
            
        elif store_clean == "faiss":
            return FAISSStore(embeddings=embeddings, persist_directory=persist_dir)
            
        else:
            logger.warning(f"Unsupported store type '{store_type}'. Defaulting to ChromaStore.")
            return ChromaStore(embeddings=embeddings, persist_directory=persist_dir)
