"""
Unit tests verifying vector store abstract implementations and managers.
"""

import pytest
from src.embeddings.factory import EmbeddingFactory
from src.vectorstores.manager import VectorStoreManager
from src.vectorstores.chroma_store import ChromaStore
from src.vectorstores.faiss_store import FAISSStore

def test_vector_store_manager_chroma():
    """
    Tests that manager returns ChromaStore wrapper.
    """
    embeddings = EmbeddingFactory.get_embeddings("mock", "dimension-384")
    store = VectorStoreManager.get_vector_store("chroma", embeddings, "./data/test_chroma")
    
    assert isinstance(store, ChromaStore)
    assert store.persist_directory == "./data/test_chroma"


def test_vector_store_manager_faiss():
    """
    Tests that manager returns FAISSStore wrapper.
    """
    embeddings = EmbeddingFactory.get_embeddings("mock", "dimension-384")
    store = VectorStoreManager.get_vector_store("faiss", embeddings, "./data/test_faiss")
    
    assert isinstance(store, FAISSStore)
    assert store.persist_directory == "./data/test_faiss"
