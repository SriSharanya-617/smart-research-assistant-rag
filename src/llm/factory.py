"""
Factory class to instantiate and wrap LLMs from multiple providers.
"""

import time
from typing import Iterator, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from src.llm.base import BaseLLM
from src.logger import setup_logger

logger = setup_logger("llm_factory")

class LangChainLLMWrapper(BaseLLM):
    """
    Wraps standard LangChain chat models to match the BaseLLM signature.
    """
    def __init__(self, lc_llm):
        self.lc_llm = lc_llm
        logger.info(f"LangChainLLMWrapper loaded for: {type(self.lc_llm).__name__}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        try:
            logger.debug("Executing LLM generation call.")
            response = self.lc_llm.invoke(messages)
            return str(response.content)
        except Exception as e:
            logger.error(f"Error during LLM invoke: {e}")
            raise RuntimeError(f"LLM execution failed: {e}")

    def stream_generate(self, prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        try:
            logger.debug("Executing streaming LLM generation call.")
            for chunk in self.lc_llm.stream(messages):
                # Standard langchain chunks return chunk objects with content
                yield str(chunk.content)
        except Exception as e:
            logger.error(f"Error during LLM streaming: {e}")
            yield f"\n[STREAM ERROR: {e}]"


class MockLLM(BaseLLM):
    """
    A Mock LLM returning pre-configured messages.
    Perfect for unit testing and local deployment checking without hitting API call limits.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        logger.info(f"MockLLM initialized simulating model: {self.model_name}")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        logger.debug(f"Mock LLM generating response for query: '{prompt[:40]}...'")
        time.sleep(0.5) # Simulate latency
        return (
            f"[Mock LLM Response - Model: {self.model_name}]\n\n"
            f"This is a mocked answer responding to your question. "
            f"In the production step, this response will be constructed by performing RAG "
            f"and asking the LLM to synthesize the retrieved source chunks."
        )

    def stream_generate(self, prompt: str, system_prompt: Optional[str] = None) -> Iterator[str]:
        logger.debug("Mock LLM starting stream generation.")
        response_text = (
            f"[Mock LLM Response - Model: {self.model_name}]\n\n"
            f"This is a simulated token stream. If you set valid API keys and change the provider "
            f"in the settings, you will see real-time inference results from your provider."
        )
        for word in response_text.split(" "):
            yield word + " "
            time.sleep(0.04)


class LLMFactory:
    """
    Factory to construct and return appropriate BaseLLM instances based on settings.
    """
    @staticmethod
    def get_llm(
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> BaseLLM:
        """
        Creates and returns a concrete LLM provider wrapper.
        
        Args:
            provider: 'openai', 'anthropic', 'ollama', 'huggingface', or 'mock'.
            model_name: Name of the model.
            api_key: Optional API authentication token.
            base_url: Optional server API URL (primarily for Ollama).
            
        Returns:
            BaseLLM: Conformed LLM wrapper instance.
        """
        provider_clean = provider.lower().strip()
        logger.info(f"Instantiating LLM model: provider={provider_clean}, model={model_name}")

        try:
            if provider_clean == "openai":
                from langchain_openai import ChatOpenAI
                lc_llm = ChatOpenAI(model=model_name, api_key=api_key)
                return LangChainLLMWrapper(lc_llm)

            elif provider_clean == "anthropic":
                from langchain_anthropic import ChatAnthropic
                lc_llm = ChatAnthropic(model_name=model_name, api_key=api_key)
                return LangChainLLMWrapper(lc_llm)

            elif provider_clean == "ollama":
                from langchain_community.chat_models import ChatOllama
                # Convert localhost URL if provided
                url = base_url or "http://localhost:11434"
                lc_llm = ChatOllama(model=model_name, base_url=url)
                return LangChainLLMWrapper(lc_llm)

            elif provider_clean == "huggingface":
                from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
                # Instantiate Endpoint first
                endpoint = HuggingFaceEndpoint(
                    repo_id=model_name,
                    huggingfacehub_api_token=api_key,
                    timeout=120
                )
                lc_llm = ChatHuggingFace(llm=endpoint)
                return LangChainLLMWrapper(lc_llm)

            elif provider_clean == "mock":
                return MockLLM(model_name)

            else:
                logger.warning(f"Unknown LLM provider '{provider}'. Falling back to MockLLM.")
                return MockLLM(model_name)

        except Exception as e:
            logger.error(f"Failed to load LLM provider '{provider}': {e}. Falling back to MockLLM.")
            return MockLLM(model_name)
