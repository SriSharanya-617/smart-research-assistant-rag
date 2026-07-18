"""
Comprehensive unit tests for the Vector Database Layer.
Verifies ChromaDB and FAISS wrapper classes, indexing, concurrency, and exception safety.
"""

import os
import pytest
import threading
import concurrent.futures
from typing import List
from src.ingestion.base import Document
from src.embeddings.factory import EmbeddingFactory
from src.vectorstores.manager import VectorStoreFactory
from src.vectorstores.chroma_store import ChromaVectorStore
from src.vectorstores.faiss_store import FAISSVectorStore
from src.vectorstores.exceptions import (
    DimensionMismatchError,
    CorruptedIndexError,
    IndexLoadError,
    VectorStoreError
)
from unittest.mock import patch
# ==========================================
# TEST FIXTURES & HELPERS
# ==========================================

@pytest.fixture
def mock_embeddings():
    # Returns MockEmbeddings with dimension=384
    return EmbeddingFactory.get_embeddings(provider="mock", model_name="test-model")


def create_sample_docs() -> List[Document]:
    return [
        Document(
            page_content="RAG search databases store vector embeddings.",
            metadata={
                "document_id": "doc-hash-1",
                "chunk_id": "doc-hash-1_chunk_0",
                "chunk_index": 0,
                "filename": "file_a.txt",
                "document_type": "txt",
                "source": "/path/file_a.txt",
                "page_number": 1,
                "ingestion_timestamp": "2026-07-18T13:30:00"
            }
        ),
        Document(
            page_content="FAISS and ChromaDB are local vector index solutions.",
            metadata={
                "document_id": "doc-hash-2",
                "chunk_id": "doc-hash-2_chunk_0",
                "chunk_index": 0,
                "filename": "file_b.pdf",
                "document_type": "pdf",
                "source": "/path/file_b.pdf",
                "page_number": 3,
                "ingestion_timestamp": "2026-07-18T13:30:00"
            }
        )
    ]

# ==========================================
# 1. CORE OPERATIONS TESTS (FAISS & CHROMA)
# ==========================================

@pytest.mark.parametrize("store_type, store_class", [
    ("faiss", FAISSVectorStore),
    ("chroma", ChromaVectorStore)
])
def test_vector_store_crud_and_search(store_type, store_class, tmp_path, mock_embeddings):
    """
    Tests standard workflow: empty index, insert, load/save, query, filter, update, delete.
    """
    persist_dir = str(tmp_path / f"test_{store_type}_index")
    
    # Initialize store
    store = VectorStoreFactory.get_vector_store(store_type, mock_embeddings, persist_dir)
    assert isinstance(store, store_class)

    # 1. Test query on Empty index (should return empty list gracefully)
    assert store.similarity_search_with_score("query", k=2) == []

    # 2. Test Add Documents
    docs = create_sample_docs()
    store.add_documents(docs)
    
    # 3. Verify stats
    stats = store.get_statistics()
    assert stats["number_of_vectors"] == 2

    # 4. Test Similarity Search with scores
    results = store.similarity_search_with_score("vector databases", k=1)
    assert len(results) == 1
    matched_doc, score = results[0]
    assert matched_doc.metadata["document_id"] in ["doc-hash-1", "doc-hash-2"]
    assert score >= 0.0

    # 5. Test Metadata Filtering
    txt_results = store.similarity_search_with_score("databases", k=2, filter={"document_type": "txt"})
    assert len(txt_results) == 1
    assert txt_results[0][0].metadata["document_type"] == "txt"

    # 6. Test Update Documents
    updated_doc = Document(
        page_content="RAG databases store dense float vector embeddings.",
        metadata=docs[0].metadata.copy()
    )
    store.update_documents([updated_doc])
    
    query_updated = store.similarity_search_with_score("vector embeddings", k=2)
    assert any("dense float" in doc.page_content for doc, _ in query_updated)

    # 7. Test Remove Documents
    store.remove_documents(["doc-hash-1_chunk_0"])
    after_remove_stats = store.get_statistics()
    assert after_remove_stats["number_of_vectors"] == 1


# ==========================================
# 2. CONCURRENCY & ROBUSTNESS TESTS
# ==========================================

def test_concurrent_reads(tmp_path, mock_embeddings):
    """
    Spawns multiple threads executing concurrent queries against the same populated index.
    """
    persist_dir = str(tmp_path / "concurrent_faiss")
    store = VectorStoreFactory.get_vector_store("faiss", mock_embeddings, persist_dir)
    store.add_documents(create_sample_docs())

    num_threads = 5
    queries = ["embeddings", "databases", "faiss", "chroma", "vectors"]

    def worker_search(query_str):
        res = store.similarity_search_with_score(query_str, k=1)
        return len(res)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker_search, q) for q in queries]
        results = [f.result() for f in futures]

    assert len(results) == num_threads
    assert all(r == 1 for r in results)


def test_corrupted_index_load(tmp_path, mock_embeddings):
    """
    Verify loading a corrupted index file structure raises CorruptedIndexError/IndexLoadError.
    """
    persist_dir = str(tmp_path / "corrupted_faiss")
    os.makedirs(persist_dir, exist_ok=True)
    
    # Write garbage bytes to the index path
    with open(os.path.join(persist_dir, "index.faiss"), "wb") as f:
        f.write(b"garbage-pickled-bytes-here-12345")
        
    store = FAISSVectorStore(mock_embeddings, persist_dir)
    
    with pytest.raises(CorruptedIndexError):
        store.load(persist_dir)


# ==========================================
# 3. VECTOR VALIDATION & ERROR TESTS
# ==========================================

def test_dimension_mismatch_validation(tmp_path, mock_embeddings):
    """
    Asserts inserting vectors of different dimension than the index raises DimensionMismatchError.
    """
    persist_dir = str(tmp_path / "mismatch_faiss")
    store = VectorStoreFactory.get_vector_store("faiss", mock_embeddings, persist_dir)
    
    # Add first valid document (sets expected dim = 384)
    docs = create_sample_docs()
    store.add_documents([docs[0]])

    # Build document with mismatched vector dimensions (e.g. 512 instead of 384)
    # We patch the embed_documents call to return a 512-dim vector
    mismatched_vector = [[0.1] * 512]
    
    with patch.object(mock_embeddings, "embed_documents", return_value=mismatched_vector):
        mismatched_doc = Document(
            page_content="Mismatched content.",
            metadata={"chunk_id": "mismatch_chunk"}
        )
        with pytest.raises(DimensionMismatchError):
            store.add_documents([mismatched_doc])


def test_invalid_vector_values(tmp_path, mock_embeddings):
    """
    Verify NaN and Infinite values are rejected during insert validation.
    """
    persist_dir = str(tmp_path / "invalid_vals_faiss")
    store = VectorStoreFactory.get_vector_store("faiss", mock_embeddings, persist_dir)
    
    # Mock return vectors containing NaN or Inf
    nan_vector = [[float("nan")] * 384]
    inf_vector = [[float("inf")] * 384]

    invalid_doc = Document(
        page_content="Invalid float values.",
        metadata={"chunk_id": "invalid_chunk"}
    )

    with patch.object(mock_embeddings, "embed_documents", return_value=nan_vector):
        with pytest.raises(ValueError) as exc_info:
            store.add_documents([invalid_doc])
        assert "contains NaN" in str(exc_info.value)

    with patch.object(mock_embeddings, "embed_documents", return_value=inf_vector):
        with pytest.raises(ValueError) as exc_info:
            store.add_documents([invalid_doc])
        assert "contains infinite" in str(exc_info.value)


@pytest.mark.parametrize("store_type,store_class", [("chroma", ChromaVectorStore), ("faiss", FAISSVectorStore)])
def test_vector_store_batch_duplicates(tmp_path, mock_embeddings, store_type, store_class):
    """
    Asserts inserting identical document chunks in the same batch filters them out without crashing.
    """
    persist_dir = str(tmp_path / f"batch_dups_{store_type}")
    store = VectorStoreFactory.get_vector_store(store_type, mock_embeddings, persist_dir)
    
    docs = create_sample_docs()
    # Create duplicate documents list
    duplicated_docs = [docs[0], docs[0], docs[0]]
    
    # This should succeed without raising expected unique ID exceptions
    store.add_documents(duplicated_docs)
    
    # Verify only one was actually added
    if store_type == "chroma":
        all_items = store.db._collection.get(include=["metadatas"])
        assert len(all_items.get("ids", [])) == 1
    else:
        assert len(store.db.docstore._dict) == 1

