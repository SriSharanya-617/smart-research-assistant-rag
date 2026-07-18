"""
Custom exceptions for the Retrieval Engine.
"""

class RetrievalError(Exception):
    """Base exception for all retrieval-related errors."""
    pass


class EmptyQueryError(RetrievalError):
    """Raised when the submitted query is empty or whitespace-only."""
    pass


class InvalidQueryError(RetrievalError):
    """Raised when the submitted query violates validation rules (e.g. length limits)."""
    pass


class VectorStoreUnavailableError(RetrievalError):
    """Raised when a query is executed but the vector database backend is not active/available."""
    pass
