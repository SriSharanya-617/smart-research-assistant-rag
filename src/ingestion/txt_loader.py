"""
Production-quality text document loader.
Supports UTF-8, UTF-8-sig (BOM), and ISO-8859-1 with size checks and pre-processing.
"""

import os
import hashlib
from typing import List, Callable, Optional
from src.ingestion.base import BaseDocumentLoader, Document
from src.ingestion.exceptions import (
    DocumentLoadError,
    UnsupportedEncodingError,
    FileLimitExceededError,
    IngestionCancelledError
)
from src.ingestion.preprocessing import TextPreprocessor
from src.logger import setup_logger
from src.config import get_config

logger = setup_logger("txt_loader")

class TXTLoader(BaseDocumentLoader):
    """
    Loads, validates, cleans, and indexes content from plain text files.
    """
    def __init__(
        self,
        source: str,
        max_file_size_mb: Optional[int] = None,
        preprocessor: Optional[TextPreprocessor] = None,
        encodings: Optional[List[str]] = None
    ):
        super().__init__(source)
        config = get_config()
        self.max_file_size_mb = max_file_size_mb or config.MAX_UPLOAD_SIZE_MB
        self.preprocessor = preprocessor or TextPreprocessor()
        self.encodings = encodings or ["utf-8", "utf-8-sig", "iso-8859-1"]

    def load(
        self,
        progress_callback: Optional[Callable[[float], None]] = None,
        cancellation_check: Optional[Callable[[], bool]] = None
    ) -> List[Document]:
        """
        Loads the TXT file and applies standard pre-processing.
        
        Args:
            progress_callback: Optional progress indicator.
            cancellation_check: Optional cancellation checker.
            
        Returns:
            List[Document]: Singleton list of parsed Document.
        """
        # 1. Validation & Size check
        if not os.path.exists(self.source):
            logger.error(f"Text file not found: {self.source}")
            raise FileNotFoundError(f"Text file not found: {self.source}")

        if not self.source.lower().endswith(".txt"):
            logger.error(f"Invalid text file extension: {self.source}")
            raise ValueError(f"Invalid file format: {self.source}. Only .txt is supported.")

        file_size_bytes = os.path.getsize(self.source)
        max_size_bytes = self.max_file_size_mb * 1024 * 1024
        if file_size_bytes > max_size_bytes:
            logger.error(f"Text file size {file_size_bytes} exceeds limit of {max_size_bytes} bytes.")
            raise FileLimitExceededError(
                f"File '{os.path.basename(self.source)}' size exceeds limit of {self.max_file_size_mb} MB."
            )

        # Check for early cancellation
        if cancellation_check and cancellation_check():
            raise IngestionCancelledError("Ingestion cancelled prior to reading text file.")

        if progress_callback:
            progress_callback(0.2)

        # 2. Read raw bytes and attempt decoding using candidate encodings
        try:
            with open(self.source, "rb") as f:
                raw_bytes = f.read()
        except Exception as e:
            logger.error(f"Error reading file {self.source}: {e}")
            raise DocumentLoadError(f"Failed to read file: {e}")

        # Check for UTF-8 BOM signature bytes
        if raw_bytes.startswith(b"\xef\xbb\xbf"):
            encodings = ["utf-8-sig", "utf-8", "iso-8859-1"]
        else:
            encodings = self.encodings
            
        content = ""
        success_encoding = None

        for enc in encodings:
            try:
                logger.debug(f"Attempting to decode {self.source} with encoding: {enc}")
                content = raw_bytes.decode(enc)
                success_encoding = enc
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if success_encoding is None:
            logger.error(f"Unsupported text file encoding for: {self.source}")
            raise UnsupportedEncodingError(
                f"Failed to decode text file '{os.path.basename(self.source)}'. "
                f"Only UTF-8, UTF-8 BOM, and ISO-8859-1 encodings are supported."
            )

        if progress_callback:
            progress_callback(0.6)

        # 3. Clean Text content
        cleaned_text = self.preprocessor.clean_text(content)
        if not cleaned_text:
            logger.warning(f"TXT document is empty after cleaning: {self.source}")
            raise DocumentLoadError(f"TXT document '{os.path.basename(self.source)}' contains no text content.")

        # Check cancellation
        if cancellation_check and cancellation_check():
            raise IngestionCancelledError("Ingestion cancelled after text decoding.")

        # 4. Generate SHA-256 document hash
        sha256 = hashlib.sha256()
        with open(self.source, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        document_id = sha256.hexdigest()

        file_name = os.path.basename(self.source)
        absolute_path = os.path.abspath(self.source)

        metadata = {
            "document_id": document_id,
            "filename": file_name,
            "page_number": 1, # TXT is a single page document
            "total_pages": 1,
            "source": absolute_path,
            "document_type": "txt",
            "encoding": success_encoding
        }

        if progress_callback:
            progress_callback(1.0)

        logger.info(f"Successfully loaded and preprocessed text document: {file_name}")
        return [Document(page_content=cleaned_text, metadata=metadata)]
