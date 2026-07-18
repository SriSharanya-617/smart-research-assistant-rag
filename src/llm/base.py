"""
Abstract base class interface for LLM model providers.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional

class BaseLLM(ABC):
    """
    Common abstraction interface for LLM text generation.
    Supports standard output and token streaming options.
    """
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Executes synchronous text generation.
        
        Args:
            prompt: User message prompt.
            system_prompt: Optional system behavior guidelines.
            
        Returns:
            str: Generated text response.
        """
        pass

    @abstractmethod
    def stream_generate(self, prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        """
        Executes real-time streaming text generation.
        
        Args:
            prompt: User message prompt.
            system_prompt: Optional system behavior guidelines.
            
        Returns:
            Iterator[str]: Token parts yielding sequentially.
        """
        pass
