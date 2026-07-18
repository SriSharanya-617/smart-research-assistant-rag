"""
Custom exceptions for the Embedding Module.
"""

class EmbeddingError(Exception):
    """Base exception for all embedding-related errors."""
    pass


class ModelLoadError(EmbeddingError):
    """Raised when an embedding model fails to download, load, or initialize."""
    pass


class EmbeddingGenerationError(EmbeddingError):
    """Raised when generating vector embeddings fails due to dimension mismatches or execution errors."""
    pass
