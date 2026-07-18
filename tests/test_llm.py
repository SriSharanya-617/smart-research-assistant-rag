"""
Unit tests verifying multi-provider LLM factory abstractions.
"""

from src.llm.factory import LLMFactory, MockLLM

def test_llm_factory_mock():
    """
    Tests that requesting a mock provider returns a MockLLM instance.
    """
    llm = LLMFactory.get_llm("mock", "gpt-4")
    assert isinstance(llm, MockLLM)
    assert llm.model_name == "gpt-4"


def test_mock_llm_generation():
    """
    Verifies that the MockLLM mock generation returns non-empty strings.
    """
    llm = LLMFactory.get_llm("mock", "gpt-4")
    response = llm.generate("Hello Assistant!")
    
    assert isinstance(response, str)
    assert "[Mock LLM Response - Model: gpt-4]" in response


def test_mock_llm_streaming():
    """
    Verifies that the MockLLM stream generation yields strings.
    """
    llm = LLMFactory.get_llm("mock", "gpt-4")
    stream = llm.stream_generate("Hello Assistant!")
    
    chunks = list(stream)
    assert len(chunks) > 0
    assert any("[Mock" in chunk for chunk in chunks)
