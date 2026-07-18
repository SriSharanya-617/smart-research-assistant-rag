"""
Unit tests for the Embeddings Module.
Verifies factory singleton behavior, lazy loading, stats tracking, normalization, and hardware fallback.
"""

import pytest
from unittest.mock import MagicMock, patch
from src.embeddings.exceptions import ModelLoadError, EmbeddingGenerationError
from src.embeddings.factory import EmbeddingFactory
from src.embeddings.sentence_transformer import SentenceTransformerProvider
from src.embeddings.factory_mock import MockEmbeddings

# ==========================================
# 1. FACTORY SINGLETON TESTS
# ==========================================

def test_factory_singleton_caching():
    """
    Verifies that subsequent factory calls return the exact same instance.
    """
    EmbeddingFactory.clear_cache()
    
    # Requesting mock provider twice
    embed1 = EmbeddingFactory.get_embeddings(provider="mock", model_name="test-model")
    embed2 = EmbeddingFactory.get_embeddings(provider="mock", model_name="test-model")
    
    # Assert they are the same singleton object
    assert embed1 is embed2
    assert isinstance(embed1, MockEmbeddings)


def test_factory_different_models():
    """
    Verifies that requesting different models returns different instances.
    """
    EmbeddingFactory.clear_cache()
    
    embed1 = EmbeddingFactory.get_embeddings(provider="mock", model_name="model-a")
    embed2 = EmbeddingFactory.get_embeddings(provider="mock", model_name="model-b")
    
    assert embed1 is not embed2


# ==========================================
# 2. SENTENCE TRANSFORMER WRAPPER TESTS
# ==========================================

@patch("src.embeddings.sentence_transformer.SentenceTransformerProvider._load_model")
def test_lazy_loading_behavior(mock_load):
    """
    Verifies that model resources are NOT loaded on initialization, only upon first query.
    """
    provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2")
    
    # Assert load has not been called yet
    mock_load.assert_not_called()
    assert provider._model is None
    
    # Trigger method that requires load
    _ = provider.get_dimension()
    mock_load.assert_called_once()


@patch("sentence_transformers.SentenceTransformer")
def test_dimension_and_embedding_mock(mock_st_class):
    """
    Tests dimension parsing and vector conversion using a mocked SentenceTransformer instance.
    """
    # Configure mock model instance
    mock_model_inst = MagicMock()
    mock_st_class.return_value = mock_model_inst
    
    # Mock dimension and encoding methods
    mock_model_inst.get_sentence_embedding_dimension.return_value = 384
    mock_model_inst.get_embedding_dimension.return_value = 384
    
    import numpy as np
    dummy_vector = [0.5] * 384
    
    # Mock encode signature matches: encode(sentences, ...)
    # SentenceTransformer.encode always returns a 2D numpy array of shape (num_sentences, dimension)
    def mock_encode(sentences, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]
        return np.array([dummy_vector for _ in sentences])
        
    mock_model_inst.encode.side_effect = mock_encode

    # Instantiate provider
    provider = SentenceTransformerProvider(model_name="BAAI/bge-small-en-v1.5")
    
    # Test dimension
    assert provider.get_dimension() == 384
    
    # Test embed query
    query_vector = provider.embed_query("Sample question.")
    assert len(query_vector) == 384
    assert query_vector[0] == 0.5
    
    # Test batch doc embed
    doc_vectors = provider.embed_documents(["Doc A", "Doc B"])
    assert len(doc_vectors) == 2
    assert len(doc_vectors[0]) == 384
    
    # Test statistics tracking
    stats = provider.get_statistics()
    assert stats["number_of_embedded_documents"] == 3  # 1 query + 2 docs
    assert stats["batch_size_used"] == 32
    assert stats["device_used"] in ["cpu", "cuda"]
    assert stats["model_name"] == "BAAI/bge-small-en-v1.5"


def test_invalid_model_error():
    """
    Verify that loading a non-existent model raises a ModelLoadError.
    """
    provider = SentenceTransformerProvider(model_name="invalid-namespace/non-existent-model")
    
    with pytest.raises(ModelLoadError) as exc_info:
        # Forcing load
        provider.embed_query("Test")
        
    assert "Failed to download or load model" in str(exc_info.value)


# ==========================================
# 3. HARDWARE DETECTION & FALLBACK TESTS
# ==========================================

def test_device_fallback_logic():
    """
    Verify that provider falls back to CPU if requested device is CUDA but CUDA is not available.
    """
    with patch("torch.cuda.is_available", return_value=False):
        provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2", device="cuda")
        # Should detect that CUDA is not available and map to CPU
        assert provider.device == "cpu"

    with patch("torch.cuda.is_available", return_value=True):
        provider = SentenceTransformerProvider(model_name="all-MiniLM-L6-v2", device="cuda")
        # Should keep CUDA if available
        assert provider.device == "cuda"
