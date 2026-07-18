"""
Document ingestion module for loading and splitting files.
"""

from src.ingestion.base import BaseDocumentLoader, Document
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.txt_loader import TXTLoader
from src.ingestion.web_loader import WebLoader
from src.ingestion.splitter import DocumentSplitter

__all__ = [
    "BaseDocumentLoader",
    "Document",
    "PDFLoader",
    "TXTLoader",
    "WebLoader",
    "DocumentSplitter"
]
