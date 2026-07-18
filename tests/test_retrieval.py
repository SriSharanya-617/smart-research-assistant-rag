"""
Unit tests for the Retrieval Engine (Phase 5).
Verifies query validation, intent categorization, caching, confidence models, and inspector outputs.
"""

import time
import pytest
from unittest.mock import MagicMock, patch
from src.ingestion.base import Document
from src.embeddings.factory import EmbeddingFactory
from src.vectorstores.faiss_store import FAISSVectorStore
from src.retrieval.exceptions import EmptyQueryError, InvalidQueryError, VectorStoreUnavailableError
from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.query_processor import QueryProcessor, QueryComplexityAnalyzer
from src.retrieval.evaluators import ConfidenceEstimator

# ==========================================
# TEST FIXTURES & HELPERS
# ==========================================

@pytest.fixture
def mock_embeddings():
    return EmbeddingFactory.get_embeddings(provider="mock", model_name="test-model")


@pytest.fixture
def populated_vector_store(tmp_path, mock_embeddings):
    persist_dir = str(tmp_path / "retrieval_test_faiss")
    store = FAISSVectorStore(mock_embeddings, persist_dir)
    
    # Populate with sample documents
    docs = [
        Document(
            page_content="Retrieval-Augmented Generation (RAG) merges LLMs with external files.",
            metadata={
                "document_id": "doc1",
                "chunk_id": "doc1_chunk0",
                "filename": "rag_overview.txt",
                "document_type": "txt",
                "source": "/docs/rag_overview.txt",
                "page_number": 1
            }
        ),
        Document(
            page_content="Vector databases store float embeddings and search them using L2 similarity metrics.",
            metadata={
                "document_id": "doc2",
                "chunk_id": "doc2_chunk0",
                "filename": "vector_db.pdf",
                "document_type": "pdf",
                "source": "/docs/vector_db.pdf",
                "page_number": 3
            }
        )
    ]
    store.add_documents(docs)
    return store

# ==========================================
# 1. QUERY PREPROCESSING TESTS
# ==========================================

def test_query_cleaner_and_normalizer():
    processor = QueryProcessor()
    
    # Unicode Normalization (NFKC compatibility equivalent)
    raw_unicode = "RAG \u2122   System"  # contains trademark sign and multiple spaces
    cleaned = processor.clean_query(raw_unicode)
    assert cleaned == "RAG TM System"

    # Whitespace normalization
    raw_spaces = "  Multiple\n   newlines\t  and spaces  "
    assert processor.clean_query(raw_spaces) == "Multiple newlines and spaces"


def test_query_validation_bounds():
    processor = QueryProcessor(max_length=50)
    
    # Blank query
    with pytest.raises(EmptyQueryError):
        processor.preprocess("   ")

    # Long query
    long_query = "a" * 55
    with pytest.raises(InvalidQueryError):
        processor.preprocess(long_query)


# ==========================================
# 2. QUERY COMPLEXITY ANALYSIS TESTS
# ==========================================

@pytest.mark.parametrize("query, expected_intent", [
    ("What is the difference between FAISS and Chroma?", "comparison request"),
    ("Summarize the main results of this research paper", "summary request"),
    ("Define Retrieval-Augmented Generation", "definition request"),
    ("Extract list of core parameters used in the benchmark", "information extraction request"),
    ("How do vector databases index embeddings?", "question")
])
def test_complexity_analyzer(query, expected_intent):
    assert QueryComplexityAnalyzer.analyze_complexity(query) == expected_intent


# ==========================================
# 3. PIPELINE SEARCH & FILTERS
# ==========================================

def test_pipeline_semantic_search_and_filters(populated_vector_store):
    pipeline = RetrievalPipeline(vector_store=populated_vector_store)
    
    # Similarity Search
    results = pipeline.retrieve("vector databases", limit=2)
    assert len(results) == 2
    assert "similarity_score" in results[0].metadata
    assert "selection_explanation" in results[0].metadata

    # Metadata filtering (PDF only)
    pdf_results = pipeline.retrieve("databases", limit=2, filter_dict={"document_type": "pdf"})
    assert len(pdf_results) == 1
    assert pdf_results[0].metadata["document_type"] == "pdf"

    # Filters returning zero results
    zero_results = pipeline.retrieve("databases", limit=2, filter_dict={"document_type": "docx"})
    assert len(zero_results) == 0


def test_pipeline_score_threshold(populated_vector_store):
    pipeline = RetrievalPipeline(vector_store=populated_vector_store)
    
    # Set threshold = -0.1 (since distance is 0.0, it is > -0.1 and will be excluded)
    results = pipeline.retrieve("vector databases", limit=2, score_threshold=-0.1)
    # Both documents should have L2 distance > -0.1 against the query in this mock corpus, resulting in 0 matches
    assert len(results) == 0


# ==========================================
# 4. CACHING & INVALIDATION
# ==========================================

def test_cache_hits_misses_and_invalidation(populated_vector_store):
    pipeline = RetrievalPipeline(vector_store=populated_vector_store)
    
    # Miss 1
    _ = pipeline.retrieve("RAG pipeline", limit=2)
    stats = pipeline.cache.get_statistics()
    assert stats["cache_hits"] == 0
    assert stats["cache_misses"] == 1

    # Hit 1
    _ = pipeline.retrieve("RAG pipeline", limit=2)
    stats = pipeline.cache.get_statistics()
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 1

    # Invalidation on DB index changes
    pipeline.invalidate_cache()
    # Miss 2 (re-search after cache invalidation)
    _ = pipeline.retrieve("RAG pipeline", limit=2)
    stats = pipeline.cache.get_statistics()
    assert stats["cache_misses"] == 2


def test_cache_ttl_expiration(populated_vector_store):
    # Set Cache TTL to 0 to simulate instantaneous expiration
    pipeline = RetrievalPipeline(vector_store=populated_vector_store, cache_ttl_seconds=0)
    
    # Query 1 (Miss 1)
    _ = pipeline.retrieve("caching test", limit=1)
    
    # Query 2 (Miss 2 - because TTL=0 has expired immediately)
    _ = pipeline.retrieve("caching test", limit=1)
    
    stats = pipeline.cache.get_statistics()
    assert stats["cache_misses"] == 2
    assert stats["cache_hits"] == 0


# ==========================================
# 5. CONFIDENCE TIERS & EXPLAINER
# ==========================================

@pytest.mark.parametrize("scores, db_type, expected_tier", [
    ([0.2, 0.3], "chromadb", "High"),       # Distance based: low distance = High confidence
    ([0.7, 0.8], "chromadb", "Medium"),
    ([1.5, 2.0], "chromadb", "Low"),
    ([0.9, 0.85], "cosine", "High"),        # Similarity based: high cosine = High confidence
    ([0.65, 0.55], "cosine", "Medium"),
    ([0.2, 0.1], "cosine", "Low")
])
def test_confidence_estimation(scores, db_type, expected_tier):
    assert ConfidenceEstimator.estimate_confidence(scores, db_type) == expected_tier


# ==========================================
# 6. RETRIEVAL INSPECTOR DIAGNOSTICS
# ==========================================

def test_retrieval_inspector_output(populated_vector_store):
    pipeline = RetrievalPipeline(vector_store=populated_vector_store)
    
    # Query
    _ = pipeline.retrieve("RAG query", limit=1, filter_dict={"document_type": "txt"})
    report = pipeline.get_inspector_report()
    
    assert report is not None
    assert "retrieval_timestamp" in report
    assert report["original_query"] == "RAG query"
    assert report["processed_query"] == "RAG query"
    assert report["applied_filters"] == {"document_type": "txt"}
    assert report["requested_k"] == 1
    assert report["returned_k"] == 1
    assert report["cache_hit"] is False
    assert "query_complexity" in report
    assert "confidence_estimate" in report
    assert "latency_seconds" in report
    assert len(report["retrieved_chunks"]) == 1
    assert "explanation" in report["retrieved_chunks"][0]
