"""
LLM and RAG Orchestration package for the Smart Research Assistant.
Exposes LLM provider classes, prompt builders, factories, and RAG pipelines.
"""

from src.llm.base import BaseLLMProvider
from src.llm.providers import (
    OpenAIProvider,
    GeminiProvider,
    OllamaProvider,
    MockLLMProvider
)
from src.llm.factory import LLMFactory
from src.llm.utils import PromptBuilder, CitationFormatter, ConversationMemory
from src.llm.rag_pipeline import RAGPipeline
from src.llm.exceptions import (
    LLMError,
    APIKeyError,
    ProviderUnavailableError,
    GenerationError
)

__all__ = [
    "BaseLLMProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "OllamaProvider",
    "MockLLMProvider",
    "LLMFactory",
    "PromptBuilder",
    "CitationFormatter",
    "ConversationMemory",
    "RAGPipeline",
    "LLMError",
    "APIKeyError",
    "ProviderUnavailableError",
    "GenerationError"
]
