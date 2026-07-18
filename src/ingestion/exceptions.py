"""
Custom exception classes for the Document Ingestion Module.
"""

class IngestionError(Exception):
    """Base exception class for all document ingestion errors."""
    pass


class DocumentLoadError(IngestionError):
    """Raised when a document cannot be loaded or parsed."""
    pass


class CorruptedDocumentError(DocumentLoadError):
    """Raised when a document format or structure is corrupted."""
    pass


class UnsupportedEncodingError(DocumentLoadError):
    """Raised when a text document's encoding is not supported or cannot be parsed."""
    pass


class WebScrapingError(DocumentLoadError):
    """Raised when a web crawling or parsing process fails."""
    pass


class TextSplittingError(IngestionError):
    """Raised when a document splitting action fails due to invalid parameters or structure."""
    pass


class FileLimitExceededError(DocumentLoadError):
    """Raised when an uploaded file exceeds the configured maximum file size."""
    pass


class IngestionCancelledError(IngestionError):
    """Raised when a long-running ingestion operation is cancelled by the caller."""
    pass
