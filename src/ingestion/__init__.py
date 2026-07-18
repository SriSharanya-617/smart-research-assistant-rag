"""
Document Ingestion Package for the Smart Research Assistant.
Exposes document loaders, splitters, text preprocessors, and custom exception classes.
"""

from src.ingestion.base import BaseDocumentLoader, Document
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.txt_loader import TXTLoader
from src.ingestion.web_loader import WebLoader
from src.ingestion.splitter import DocumentSplitter
from src.ingestion.preprocessing import TextPreprocessor
from src.ingestion.exceptions import (
    IngestionError,
    DocumentLoadError,
    CorruptedDocumentError,
    UnsupportedEncodingError,
    WebScrapingError,
    TextSplittingError,
    FileLimitExceededError,
    IngestionCancelledError
)

__all__ = [
    "BaseDocumentLoader",
    "Document",
    "PDFLoader",
    "TXTLoader",
    "WebLoader",
    "DocumentSplitter",
    "TextPreprocessor",
    "IngestionError",
    "DocumentLoadError",
    "CorruptedDocumentError",
    "UnsupportedEncodingError",
    "WebScrapingError",
    "TextSplittingError",
    "FileLimitExceededError",
    "IngestionCancelledError"
]
