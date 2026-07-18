"""
Unit tests for the LLM Integration & RAG Pipeline Module (Phase 6).
Verifies PromptBuilder, CitationFormatter, ConversationMemory, and RAGPipeline execution.
"""

import pytest
from unittest.mock import MagicMock
from src.ingestion.base import Document
from src.retrieval.pipeline import RetrievalPipeline
from src.llm.exceptions import APIKeyError, GenerationError
from src.llm.factory import LLMFactory
from src.llm.providers import MockLLMProvider
from src.llm.utils import PromptBuilder, CitationFormatter, ConversationMemory
from src.embeddings.factory import EmbeddingFactory
from src.llm.rag_pipeline import RAGPipeline

# ==========================================
# TEST FIXTURES & HELPERS
# ==========================================

@pytest.fixture
def mock_embeddings():
    return EmbeddingFactory.get_embeddings(provider="mock", model_name="test-model")


@pytest.fixture
def populated_vector_store(tmp_path, mock_embeddings):
    from src.vectorstores.faiss_store import FAISSVectorStore
    persist_dir = str(tmp_path / "rag_test_faiss")
    store = FAISSVectorStore(mock_embeddings, persist_dir)
    docs = create_mock_documents()
    store.add_documents(docs)
    return store


def create_mock_documents() -> list[Document]:
    return [
        Document(
            page_content="Retrieval-Augmented Generation (RAG) is a technique that binds models to knowledge databases.",
            metadata={
                "document_id": "sha-doc-1",
                "chunk_id": "sha-doc-1_chunk_0",
                "filename": "rag_basics.txt",
                "page_number": 1,
                "document_type": "txt",
                "source": "/docs/rag_basics.txt"
            }
        ),
        Document(
            page_content="Chunking documents prevents LLMs from exceeding token limits.",
            metadata={
                "document_id": "sha-doc-2",
                "chunk_id": "sha-doc-2_chunk_0",
                "filename": "chunking_guide.pdf",
                "page_number": 5,
                "document_type": "pdf",
                "source": "/docs/chunking_guide.pdf"
            }
        )
    ]

# ==========================================
# 1. PROMPT BUILDER TESTS
# ==========================================

def test_prompt_builder_templates():
    builder = PromptBuilder()
    docs = create_mock_documents()
    
    # 1. QA Prompt
    qa_prompt = builder.build_prompt("How does chunking work?", docs, intent="question")
    assert "=== RETRIEVED CONTEXT ===" in qa_prompt
    assert "=== USER QUESTION ===" in qa_prompt
    assert "chunking_guide.pdf" in qa_prompt
    assert "How does chunking work?" in qa_prompt

    # 2. Summarization Prompt
    sum_prompt = builder.build_prompt("Summarize the text", docs, intent="summary request")
    assert "Provide a clear summary" in sum_prompt


# ==========================================
# 2. CONVERSATION MEMORY TESTS
# ==========================================

def test_conversation_memory_sliding_window():
    memory = ConversationMemory(max_exchanges=2, enabled=True)
    
    # Add Exchange 1
    memory.add_exchange("Hello", "Hi there")
    assert len(memory.get_history()) == 2
    
    # Add Exchange 2
    memory.add_exchange("What is RAG?", "RAG combines searches with models.")
    assert len(memory.get_history()) == 4
    
    # Add Exchange 3 (should evict exchange 1, leaving exchanges 2 and 3)
    memory.add_exchange("How does caching work?", "It stores query hits.")
    history = memory.get_history()
    assert len(history) == 4
    assert "Hello" not in history[0]
    assert "What is RAG?" in history[0]


def test_conversation_memory_disabled():
    memory = ConversationMemory(max_exchanges=5, enabled=False)
    memory.add_exchange("Query", "Response")
    assert len(memory.get_history()) == 0


# ==========================================
# 3. CITATION FORMATTER TESTS
# ==========================================

def test_citation_formatter_deduplication():
    docs = create_mock_documents()
    
    # Add a duplicate document to verify deduplication
    duplicate_docs = docs + [docs[0]]
    
    citation_text = CitationFormatter.format_citations(duplicate_docs)
    assert citation_text.count("rag_basics.txt") == 1
    assert "sha-doc-1" in citation_text
    assert "sha-doc-2" in citation_text


def test_citation_formatter_missing_metadata():
    doc = Document(
        page_content="Text without file metadata.",
        metadata={
            "document_id": "sha-doc-3",
            "chunk_id": "sha-doc-3_chunk_0"
        }
    )
    citation_text = CitationFormatter.format_citations([doc])
    assert "Unknown File" in citation_text
    assert "Unknown Page" in citation_text


# ==========================================
# 4. RAG PIPELINE EXECUTION TESTS
# ==========================================

def test_rag_pipeline_success(populated_vector_store):
    from src.retrieval.pipeline import RetrievalPipeline
    
    # Initialize mock components
    mock_retrieval = RetrievalPipeline(vector_store=populated_vector_store)
    mock_llm = MockLLMProvider(model_name="mock-gpt")
    
    pipeline = RAGPipeline(
        retrieval_pipeline=mock_retrieval,
        llm_provider=mock_llm,
        memory_enabled=True
    )
    
    # Query RAG
    answer = pipeline.answer_question("What is RAG?")
    assert "Mock response" in answer
    assert "=== SOURCES & CITATIONS ===" in answer
    
    # Check execution metadata
    metadata = pipeline.get_last_run_metadata()
    assert metadata["retrieved_count"] > 0
    assert metadata["llm_metadata"]["model_name"] == "mock-gpt"
    assert metadata["confidence_estimate"] in ["High", "Medium", "Low"]


def test_rag_pipeline_empty_context():
    # Setup mock retriever that returns empty list
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = []
    mock_retrieval.get_inspector_report.return_value = {"confidence_estimate": "Low", "cache_hit": False}
    
    mock_llm = MockLLMProvider(model_name="mock-gpt")
    
    pipeline = RAGPipeline(
        retrieval_pipeline=mock_retrieval,
        llm_provider=mock_llm,
        safe_fallback_response="Fallback response."
    )
    
    # Query should return fallback immediately without calling LLM
    answer = pipeline.answer_question("Unknown query topic")
    assert answer == "Fallback response."
    
    metadata = pipeline.get_last_run_metadata()
    assert metadata["retrieved_count"] == 0
    assert metadata["confidence_estimate"] == "Low"


# ==========================================
# 5. CONFIGURATION & PROVIDER FACTORY TESTS
# ==========================================

def test_provider_switching_and_factory():
    LLMFactory.clear_cache()
    
    # Get initial Mock provider
    prov1 = LLMFactory.get_llm_provider(provider="mock", model_name="gpt-1")
    prov2 = LLMFactory.get_llm_provider(provider="mock", model_name="gpt-1")
    
    # Verify singleton caching works
    assert prov1 is prov2
    
    # Swap provider to another simulated model
    prov3 = LLMFactory.get_llm_provider(provider="mock", model_name="gpt-2")
    assert prov1 is not prov3
