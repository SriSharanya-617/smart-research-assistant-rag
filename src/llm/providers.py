"""
Concrete LLM Provider integrations wrapping OpenAI, Gemini, Ollama, and Mock backends.
Tracks token counts, execution latencies, and retries.
"""

import time
import os
import logging
from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from src.llm.base import BaseLLMProvider
from src.llm.exceptions import APIKeyError, ProviderUnavailableError, GenerationError
from src.logger import setup_logger

logger = setup_logger("llm_providers")

class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI Chat model wrapper.
    """
    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 1000, api_key: Optional[str] = None):
        super().__init__(model_name, temperature, max_tokens)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            raise APIKeyError("OpenAI API Key is missing. Set OPENAI_API_KEY environment variable.")
            
        from langchain_openai import ChatOpenAI
        
        # Configure model
        chat = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            max_retries=3
        )
        
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        start_time = time.time()
        try:
            response = chat.invoke(messages)
            latency = time.time() - start_time
            
            # Extract token metrics
            token_usage = None
            if response.response_metadata and "token_usage" in response.response_metadata:
                token_usage = response.response_metadata["token_usage"]
                
            self.last_metadata = {
                "provider_name": "openai",
                "model_name": self.model_name,
                "generation_latency": latency,
                "retry_count": 0,  # Managed internally by LangChain client retries
                "token_usage": token_usage
            }
            
            return response.content
            
        except Exception as e:
            logger.error(f"OpenAI Generation failure: {e}")
            if "api_key" in str(e).lower() or "auth" in str(e).lower() or "401" in str(e):
                raise APIKeyError(f"OpenAI authentication failed: {e}")
            raise ProviderUnavailableError(f"OpenAI service request failed: {e}")


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini Chat model wrapper.
    """
    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 1000, api_key: Optional[str] = None):
        super().__init__(model_name, temperature, max_tokens)
        # Gemini uses GOOGLE_API_KEY or GEMINI_API_KEY
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            raise APIKeyError("Gemini API Key is missing. Set GOOGLE_API_KEY environment variable.")
            
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        chat = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            google_api_key=self.api_key
        )
        
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        start_time = time.time()
        try:
            response = chat.invoke(messages)
            latency = time.time() - start_time
            
            self.last_metadata = {
                "provider_name": "gemini",
                "model_name": self.model_name,
                "generation_latency": latency,
                "retry_count": 0,
                "token_usage": None  # Google GenAI sometimes omits token totals in invoke metadata
            }
            
            return response.content
            
        except Exception as e:
            logger.error(f"Gemini generation failure: {e}")
            if "api key" in str(e).lower() or "key not found" in str(e).lower() or "400" in str(e):
                raise APIKeyError(f"Gemini authentication failed: {e}")
            raise ProviderUnavailableError(f"Gemini service request failed: {e}")


class OllamaProvider(BaseLLMProvider):
    """
    Ollama local models wrapper.
    """
    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 1000, base_url: str = "http://localhost:11434"):
        super().__init__(model_name, temperature, max_tokens)
        self.base_url = base_url

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        from langchain_community.chat_models import ChatOllama
        
        chat = ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            base_url=self.base_url
        )
        
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        start_time = time.time()
        try:
            response = chat.invoke(messages)
            latency = time.time() - start_time
            
            self.last_metadata = {
                "provider_name": "ollama",
                "model_name": self.model_name,
                "generation_latency": latency,
                "retry_count": 0,
                "token_usage": None
            }
            
            return response.content
            
        except Exception as e:
            logger.error(f"Ollama local generation failure: {e}")
            raise ProviderUnavailableError(f"Failed to communicate with local Ollama service: {e}")


class MockLLMProvider(BaseLLMProvider):
    """
    Mock LLM generator for testing and offline environments.
    """
    def __init__(self, model_name: str = "mock-model", temperature: float = 0.0, max_tokens: int = 1000, force_fail: bool = False):
        super().__init__(model_name, temperature, max_tokens)
        self.force_fail = force_fail

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if self.force_fail:
            raise GenerationError("Simulated LLM generation error.")
            
        start_time = time.time()
        # Mock answers based on query prompts
        prompt_lower = prompt.lower()
        if "compare" in prompt_lower or "difference" in prompt_lower:
            text = "Mock comparison answer based on context."
        elif "summarize" in prompt_lower:
            text = "Mock summarization overview based on context."
        else:
            text = "Mock response answering the user query."

        latency = time.time() - start_time
        self.last_metadata = {
            "provider_name": "mock",
            "model_name": self.model_name,
            "generation_latency": latency,
            "retry_count": 0,
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25}
        }
        return text
