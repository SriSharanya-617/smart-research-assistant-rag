"""
Abstract base class interface for LLM Providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Generator, Optional, Union

class BaseLLMProvider(ABC):
    """
    Common abstraction layer interface for text generation providers.
    Supports standard generation and is extensible for token streaming.
    """
    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 1000):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Diagnostics metadata for last execution run
        self.last_metadata: Dict[str, Any] = {
            "provider_name": self.__class__.__name__,
            "model_name": self.model_name,
            "generation_latency": 0.0,
            "retry_count": 0,
            "token_usage": None
        }

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Submits prompt to LLM and returns the completed text response.
        
        Args:
            prompt: Generated user query + context prompt.
            system_prompt: Optional system instructions override.
            
        Returns:
            str: Generated text completions.
        """
        pass

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Optional token streaming generator. Defaults to a single chunk output.
        Allows adding future streaming UI capabilities without breaking the public API.
        """
        yield self.generate(prompt, system_prompt=system_prompt)

    def get_last_metadata(self) -> Dict[str, Any]:
        """
        Exposes generation execution metrics.
        """
        return self.last_metadata
