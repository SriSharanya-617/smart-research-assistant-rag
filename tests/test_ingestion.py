"""
Unit tests for document parsing loaders and text splitting.
"""

from src.ingestion.base import Document
from src.ingestion.splitter import DocumentSplitter

def test_document_splitter_chunking():
    """
    Checks that long documents are split into smaller chunks.
    """
    raw_doc = Document(
        page_content="This is page 1. " * 100, # Large string
        metadata={"source": "test_file.txt"}
    )
    
    splitter = DocumentSplitter(chunk_size=100, chunk_overlap=20)
    chunked_docs = splitter.split_documents([raw_doc])
    
    assert len(chunked_docs) > 1
    assert chunked_docs[0].metadata["source"] == "test_file.txt"
    assert "chunk_index" in chunked_docs[0].metadata
    assert "total_chunks" in chunked_docs[0].metadata
    assert chunked_docs[0].metadata["total_chunks"] == len(chunked_docs)
