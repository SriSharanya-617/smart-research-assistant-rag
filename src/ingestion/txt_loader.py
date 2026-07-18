"""
Text document loader module.
"""

import os
from typing import List
from src.ingestion.base import BaseDocumentLoader, Document
from src.logger import setup_logger

logger = setup_logger("txt_loader")

class TXTLoader(BaseDocumentLoader):
    """
    Loads text content from a plain text file.
    """
    def load(self) -> List[Document]:
        """
        Reads the TXT file and returns a list containing a single Document object.
        
        Returns:
            List[Document]: List of documents (typically one representing the whole file).
        """
        if not os.path.exists(self.source):
            logger.error(f"Text file not found at: {self.source}")
            raise FileNotFoundError(f"File not found: {self.source}")

        try:
            logger.info(f"Opening text file for parsing: {self.source}")
            with open(self.source, "r", encoding="utf-8") as f:
                content = f.read()
            
            file_name = os.path.basename(self.source)
            metadata = {
                "source": self.source,
                "file_name": file_name,
                "file_size_bytes": os.path.getsize(self.source)
            }
            
            logger.info(f"Successfully loaded text file: {file_name}")
            return [Document(page_content=content.strip(), metadata=metadata)]
            
        except Exception as e:
            logger.error(f"Error loading TXT from {self.source}: {e}")
            raise RuntimeError(f"Failed to load TXT: {e}")
