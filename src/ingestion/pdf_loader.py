"""
Production-quality PDF document loader utilizing the pypdf library.
Extensively validates files, handles exceptions, parses metadata, and supports callbacks.
"""

import os
import hashlib
from typing import List, Callable, Optional
from pypdf import PdfReader
from src.ingestion.base import BaseDocumentLoader, Document
from src.ingestion.exceptions import (
    IngestionError,
    DocumentLoadError,
    CorruptedDocumentError,
    FileLimitExceededError,
    IngestionCancelledError
)
from src.ingestion.preprocessing import TextPreprocessor
from src.logger import setup_logger
from src.config import get_config

logger = setup_logger("pdf_loader")

class PDFLoader(BaseDocumentLoader):
    """
    Loads, validates, and parses text page-by-page from a PDF file.
    """
    def __init__(
        self,
        source: str,
        max_file_size_mb: Optional[int] = None,
        preprocessor: Optional[TextPreprocessor] = None
    ):
        super().__init__(source)
        config = get_config()
        self.max_file_size_mb = max_file_size_mb or config.MAX_UPLOAD_SIZE_MB
        self.preprocessor = preprocessor or TextPreprocessor()

    def load(
        self,
        progress_callback: Optional[Callable[[float], None]] = None,
        cancellation_check: Optional[Callable[[], bool]] = None
    ) -> List[Document]:
        """
        Parses PDF, extracting text and detailed metadata.
        
        Args:
            progress_callback: Optional function called with progress (0.0 to 1.0).
            cancellation_check: Optional function returning True if operation should cancel.
            
        Returns:
            List[Document]: List of parsed Documents.
        """
        # 1. Validation & Size check
        if not os.path.exists(self.source):
            logger.error(f"PDF file not found: {self.source}")
            raise FileNotFoundError(f"PDF file not found: {self.source}")
            
        if not self.source.lower().endswith(".pdf"):
            logger.error(f"Invalid file extension: {self.source}")
            raise ValueError(f"Invalid file format: {self.source}. Only .pdf is supported.")

        file_size_bytes = os.path.getsize(self.source)
        max_size_bytes = self.max_file_size_mb * 1024 * 1024
        if file_size_bytes > max_size_bytes:
            logger.error(f"File size {file_size_bytes} exceeds limit of {max_size_bytes} bytes.")
            raise FileLimitExceededError(
                f"File '{os.path.basename(self.source)}' size exceeds limit of {self.max_file_size_mb} MB."
            )

        documents: List[Document] = []
        
        try:
            logger.info(f"Opening PDF file: {self.source}")
            reader = PdfReader(self.source)
            total_pages = len(reader.pages)
            
            if total_pages == 0:
                logger.error(f"Corrupted or empty PDF: {self.source} has 0 pages.")
                raise CorruptedDocumentError(f"PDF document '{self.source}' has 0 pages.")

            # Check if encrypted
            if reader.is_encrypted:
                logger.warning(f"PDF is encrypted: {self.source}. Attempting to decrypt...")
                try:
                    # Attempt decrypting with empty password
                    success = reader.decrypt("")
                    if not success:
                        raise DocumentLoadError("PDF is password-encrypted and cannot be parsed without password.")
                except Exception as e:
                    logger.error(f"Decryption failed: {e}")
                    raise DocumentLoadError(f"Encrypted PDF load failed: {e}")

            # Extract Document Info
            doc_info = reader.metadata or {}
            author = doc_info.get("/Author")
            title = doc_info.get("/Title")
            creation_date = doc_info.get("/CreationDate")
            
            # Convert values to clean strings
            author_str = str(author).strip() if author else None
            title_str = str(title).strip() if title else None
            creation_date_str = str(creation_date).strip() if creation_date else None

            # Calculate deterministic file hash (SHA-256) of raw content bytes
            sha256 = hashlib.sha256()
            with open(self.source, "rb") as f:
                while chunk := f.read(8192):
                    sha256.update(chunk)
            document_id = sha256.hexdigest()

            # Process pages
            accumulated_text_len = 0
            file_name = os.path.basename(self.source)
            absolute_path = os.path.abspath(self.source)

            for page_num in range(total_pages):
                # Check for cancellation
                if cancellation_check and cancellation_check():
                    logger.warning("Ingestion cancelled during PDF page loop.")
                    raise IngestionCancelledError("Ingestion cancelled by caller during PDF page reading.")

                # Extract text
                page = reader.pages[page_num]
                raw_text = page.extract_text() or ""
                
                # Apply text cleaning
                cleaned_text = self.preprocessor.clean_text(raw_text)
                accumulated_text_len += len(cleaned_text)

                if cleaned_text:
                    metadata = {
                        "document_id": document_id,
                        "filename": file_name,
                        "page_number": page_num + 1,
                        "total_pages": total_pages,
                        "source": absolute_path,
                        "document_type": "pdf",
                        "author": author_str,
                        "title": title_str,
                        "creation_date": creation_date_str,
                    }
                    documents.append(Document(page_content=cleaned_text, metadata=metadata))

                # Progress callback
                if progress_callback:
                    progress_callback((page_num + 1) / total_pages)

            # Check if empty (or scanned PDF with no text extracted)
            if accumulated_text_len == 0:
                logger.error(f"Scanned or empty PDF: no text extracted from {self.source}")
                raise DocumentLoadError(
                    f"PDF document '{file_name}' contains no extractable text. "
                    "Scanned documents are not supported."
                )

            logger.info(f"Successfully processed {len(documents)} text-containing pages from {file_name}")
            
        except IngestionError:
            # Re-raise custom ingestion errors directly
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading PDF from {self.source}: {e}")
            raise CorruptedDocumentError(f"Corrupted or invalid PDF format: {e}")

        return documents
