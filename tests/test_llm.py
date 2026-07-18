"""
Unit tests verifying multi-provider LLM factory abstractions.
"""

from src.llm.factory import LLMFactory
from src.llm.providers import MockLLMProvider

def test_llm_factory_mock():
    """
    Tests that requesting a mock provider returns a MockLLMProvider instance.
    """
    # LLMFactory exposes get_llm_provider
    llm = LLMFactory.get_llm_provider("mock", "gpt-4")
    assert isinstance(llm, MockLLMProvider)
    assert llm.model_name == "gpt-4"


def test_mock_llm_generation():
    """
    Verifies that the MockLLMProvider mock generation returns non-empty strings.
    """
    llm = LLMFactory.get_llm_provider("mock", "gpt-4")
    response = llm.generate("Hello Assistant!")
    
    assert isinstance(response, str)
    assert "Mock response answering the user query." in response or "Mock" in response


def test_mock_llm_streaming():
    """
    Verifies that the MockLLMProvider stream generation yields strings.
    """
    llm = LLMFactory.get_llm_provider("mock", "gpt-4")
    stream = llm.generate_stream("Hello Assistant!")
    
    chunks = list(stream)
    assert len(chunks) > 0
    assert any("Mock" in chunk for chunk in chunks)
