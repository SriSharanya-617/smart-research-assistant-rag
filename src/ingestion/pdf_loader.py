"""
PDF document loader wrapper around pypdf library.
"""

import os
from typing import List
from pypdf import PdfReader
from src.ingestion.base import BaseDocumentLoader, Document
from src.logger import setup_logger

logger = setup_logger("pdf_loader")

class PDFLoader(BaseDocumentLoader):
    """
    Loads text content from a PDF file using pypdf.
    """
    def load(self) -> List[Document]:
        """
        Reads the PDF and returns a list of Document objects (one per page).
        
        Returns:
            List[Document]: List of documents representing each page of the PDF.
        """
        if not os.path.exists(self.source):
            logger.error(f"PDF file not found at: {self.source}")
            raise FileNotFoundError(f"File not found: {self.source}")

        documents: List[Document] = []
        try:
            logger.info(f"Opening PDF file for parsing: {self.source}")
            reader = PdfReader(self.source)
            file_name = os.path.basename(self.source)
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                cleaned_text = text.strip()
                if cleaned_text:
                    metadata = {
                        "source": self.source,
                        "file_name": file_name,
                        "page": page_num + 1,
                        "total_pages": len(reader.pages)
                    }
                    documents.append(Document(page_content=cleaned_text, metadata=metadata))
            
            logger.info(f"Successfully extracted {len(documents)} pages from {file_name}")
            
        except Exception as e:
            logger.error(f"Error loading PDF from {self.source}: {e}")
            raise RuntimeError(f"Failed to load PDF: {e}")

        return documents
