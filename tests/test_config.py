"""
Unit tests for configuration loaders and settings.
"""

import pytest
from src.config import AppConfig

def test_default_config_values():
    """
    Verifies that default config initialization carries fallback constants.
    """
    config = AppConfig()
    assert config.LLM_PROVIDER in ["openai", "anthropic", "ollama", "huggingface"]
    assert config.EMBEDDING_PROVIDER in ["huggingface", "openai"]
    assert config.VECTOR_STORE_TYPE in ["chroma", "faiss"]


def test_invalid_llm_provider():
    """
    Asserts that AppConfig throws validation errors on invalid providers.
    """
    with pytest.raises(ValueError):
        AppConfig(LLM_PROVIDER="unsupported_provider")


def test_invalid_vector_store():
    """
    Asserts that AppConfig throws validation errors on invalid vector stores.
    """
    with pytest.raises(ValueError):
        AppConfig(VECTOR_STORE_TYPE="postgres")
