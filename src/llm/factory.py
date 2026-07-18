"""
LLM Factory registry for instantiating and caching LLM providers.
"""

from typing import Dict, Tuple, Optional
from src.llm.base import BaseLLMProvider
from src.llm.providers import OpenAIProvider, GeminiProvider, OllamaProvider, MockLLMProvider
from src.llm.exceptions import LLMError
from src.logger import setup_logger
from src.config import get_config

logger = setup_logger("llm_factory")

# Global singleton registry cache
_llm_cache: Dict[Tuple[str, str], BaseLLMProvider] = {}

class LLMFactory:
    """
    Factory constructor managing loaded LLM providers.
    Uses caches to avoid creating redundant client instances.
    """
    @staticmethod
    def get_llm_provider(
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        api_key: Optional[str] = None
    ) -> BaseLLMProvider:
        """
        Retrieves or initializes an LLM provider singleton wrapper.
        
        Args:
            provider: 'openai', 'gemini', 'ollama', or 'mock'. Defaults to config.
            model_name: Model ID string. Defaults to config.
            temperature: Sampling temperature.
            max_tokens: Limit on generated tokens.
            api_key: API auth key override.
            
        Returns:
            BaseLLMProvider: The instantiated provider singleton wrapper.
        """
        config = get_config()
        
        # Load from config defaults if parameters are omitted
        prov_clean = (provider or config.LLM_PROVIDER).lower().strip()
        model_clean = (model_name or config.LLM_MODEL).strip()
        
        cache_key = (prov_clean, model_clean)
        
        # Check cache registry
        if cache_key in _llm_cache:
            logger.info(f"Retrieving cached LLM provider singleton for key: {cache_key}")
            return _llm_cache[cache_key]

        logger.info(f"Creating new LLM provider instance for key: {cache_key}")
        
        try:
            if prov_clean == "openai":
                provider_inst = OpenAIProvider(
                    model_name=model_clean,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key
                )
            elif prov_clean in ["gemini", "google"]:
                provider_inst = GeminiProvider(
                    model_name=model_clean,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key
                )
            elif prov_clean == "ollama":
                # Check config url base if available
                base_url = getattr(config, "OLLAMA_BASE_URL", "http://localhost:11434")
                provider_inst = OllamaProvider(
                    model_name=model_clean,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    base_url=base_url
                )
            elif prov_clean == "mock":
                provider_inst = MockLLMProvider(
                    model_name=model_clean,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                logger.warning(
                    f"Unsupported LLM provider '{provider}'. "
                    f"Defaulting to MockLLMProvider."
                )
                provider_inst = MockLLMProvider(
                    model_name=model_clean,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
            # Cache the instance
            _llm_cache[cache_key] = provider_inst
            return provider_inst
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider: {e}")
            raise LLMError(f"Factory could not construct provider: {e}")

    @staticmethod
    def clear_cache() -> None:
        """
        Clears the cached LLM providers.
        """
        global _llm_cache
        logger.info("Clearing LLM providers cache.")
        _llm_cache.clear()
