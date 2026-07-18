"""
Custom exceptions for the Vector Store Layer.
"""

class VectorStoreError(Exception):
    """Base exception class for all vector store errors."""
    pass


class IndexLoadError(VectorStoreError):
    """Raised when a vector store index fails to load from disk."""
    pass


class DimensionMismatchError(VectorStoreError):
    """Raised when the input vector dimension does not match the store index dimension."""
    pass


class DuplicateVectorError(VectorStoreError):
    """Raised when an attempt is made to insert an already existing vector/chunk ID."""
    pass


class CorruptedIndexError(VectorStoreError):
    """Raised when a saved vector index file is corrupted or unreadable."""
    pass
