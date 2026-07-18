"""
Document splitter wrapping LangChain's text splitters to slice Documents.
"""

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.ingestion.base import Document
from src.logger import setup_logger

logger = setup_logger("document_splitter")

class DocumentSplitter:
    """
    Slices raw loaded Documents into standard manageable chunks.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Initialize LangChain's recursive splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Slices a list of incoming Documents into smaller text chunks.
        
        Args:
            documents: List of raw parsed Documents.
            
        Returns:
            List[Document]: List of chunked Documents.
        """
        logger.info(f"Splitting {len(documents)} source documents (size={self.chunk_size}, overlap={self.chunk_overlap})")
        
        chunked_docs: List[Document] = []
        for doc in documents:
            # Generate raw chunks from the content string
            chunks = self.splitter.split_text(doc.page_content)
            
            for idx, chunk in enumerate(chunks):
                # Duplicate the source metadata and append chunk specific info
                chunk_metadata = doc.metadata.copy()
                chunk_metadata["chunk_index"] = idx
                chunk_metadata["total_chunks"] = len(chunks)
                
                chunked_docs.append(
                    Document(page_content=chunk, metadata=chunk_metadata)
                )
                
        logger.info(f"Split completed. Created {len(chunked_docs)} chunked documents.")
        return chunked_docs
