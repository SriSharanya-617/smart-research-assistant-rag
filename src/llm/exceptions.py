"""
Custom exceptions for the LLM Module.
"""

class LLMError(Exception):
    """Base exception for all LLM and generation errors."""
    pass


class APIKeyError(LLMError):
    """Raised when there is an authentication failure or invalid API key."""
    pass


class ProviderUnavailableError(LLMError):
    """Raised when the requested model provider is offline, timed out, or inaccessible."""
    pass


class GenerationError(LLMError):
    """Raised when token generation fails or content violation triggers occur."""
    pass
