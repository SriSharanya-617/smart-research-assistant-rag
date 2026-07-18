"""
Text splitter wrapping LangChain's RecursiveCharacterTextSplitter.
Enforces metadata consistency, unique chunk ID generation, and timestamps.
"""

import datetime
from typing import List, Callable, Optional
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.ingestion.base import Document
from src.ingestion.exceptions import TextSplittingError, IngestionCancelledError
from src.logger import setup_logger

logger = setup_logger("document_splitter")

class DocumentSplitter:
    """
    Slices raw parsed Documents into chunks, propagating parent metadata and stamps.
    """
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]
        
        # Configure underlying LangChain splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            keep_separator=True
        )

    def split_documents(
        self,
        documents: List[Document],
        progress_callback: Optional[Callable[[float], None]] = None,
        cancellation_check: Optional[Callable[[], bool]] = None
    ) -> List[Document]:
        """
        Splits a list of raw parsed Documents into smaller chunks.
        
        Args:
            documents: List of raw parsed Documents.
            progress_callback: Optional progress indicator.
            cancellation_check: Optional cancellation checker.
            
        Returns:
            List[Document]: List of chunked Documents with metadata.
        """
        logger.info(f"Initiating splitting for {len(documents)} source documents.")
        
        if not documents:
            return []

        chunked_docs: List[Document] = []
        total_docs = len(documents)

        # Record ingestion timestamp once for the batch
        ingestion_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        try:
            for idx, doc in enumerate(documents):
                # Check cancellation inside loop
                if cancellation_check and cancellation_check():
                    logger.warning("Text splitting cancelled by caller.")
                    raise IngestionCancelledError("Ingestion cancelled by caller during text splitting.")

                # Extract basic info
                doc_id = doc.metadata.get("document_id")
                if not doc_id:
                    raise TextSplittingError("Cannot split document: missing 'document_id' metadata.")

                filename = doc.metadata.get("filename", "Unknown")
                doc_type = doc.metadata.get("document_type", "txt")
                source = doc.metadata.get("source", "Unknown")
                page_number = doc.metadata.get("page_number", 1)

                # Split text content
                chunks = self.splitter.split_text(doc.page_content)
                total_chunks = len(chunks)

                for c_idx, chunk in enumerate(chunks):
                    # Unique chunk ID combining document hash and chunk offset
                    chunk_id = f"{doc_id}_chunk_{c_idx}"
                    
                    chunk_metadata = {
                        "document_id": doc_id,
                        "chunk_id": chunk_id,
                        "chunk_index": c_idx,
                        "total_chunks": total_chunks,
                        "filename": filename,
                        "document_type": doc_type,
                        "source": source,
                        "page_number": page_number,
                        "ingestion_timestamp": ingestion_timestamp
                    }
                    
                    # Propagate extra metadata fields (e.g. author, creation date, domain)
                    for key, val in doc.metadata.items():
                        if key not in chunk_metadata:
                            chunk_metadata[key] = val

                    chunked_docs.append(Document(page_content=chunk, metadata=chunk_metadata))

                # Update progress
                if progress_callback:
                    progress_callback((idx + 1) / total_docs)

            # Performance reporting logs
            total_generated_chunks = len(chunked_docs)
            avg_chunk_len = sum(len(c.page_content) for c in chunked_docs) / max(1, total_generated_chunks)
            
            logger.info(
                f"Text splitting completed. Documents={total_docs} -> Chunks={total_generated_chunks}. "
                f"Average chunk length={avg_chunk_len:.1f} characters."
            )
            
        except IngestionCancelledError:
            raise
        except Exception as e:
            logger.error(f"Error during text splitting: {e}")
            raise TextSplittingError(f"Failed to split documents: {e}")

        return chunked_docs
