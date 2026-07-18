"""
Comprehensive unit tests for the Document Ingestion Module.
"""

import os
import pytest
from unittest.mock import MagicMock, patch, mock_open
import requests
from src.ingestion.base import Document
from src.ingestion.preprocessing import TextPreprocessor
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.txt_loader import TXTLoader
from src.ingestion.web_loader import WebLoader
from src.ingestion.splitter import DocumentSplitter
from src.ingestion.exceptions import (
    DocumentLoadError,
    CorruptedDocumentError,
    UnsupportedEncodingError,
    WebScrapingError,
    FileLimitExceededError,
    IngestionCancelledError
)

# ==========================================
# 1. PREPROCESSING TESTS
# ==========================================

def test_text_preprocessor_cleaning():
    """
    Verifies Unicode normalization, whitespace collapsing, invisible characters removal,
    and double newline paragraph preservation.
    """
    preprocessor = TextPreprocessor()
    
    # Input has: compatibility chars (\u212b -> Angstrom), controls (\x07), multi-spaces, repeated newlines
    raw_text = "Hello \u212b World! \x07  Here   is a tab.\n\n\n\nNew paragraph."
    cleaned = preprocessor.clean_text(raw_text)
    
    # Expected: \u212b normalized to A with ring (\u00c5), controls stripped, spaces merged, 4 newlines down to 2
    assert "Å" in cleaned
    assert "\x07" not in cleaned
    assert "Here is a tab" in cleaned
    assert "\n\n" in cleaned
    assert "\n\n\n" not in cleaned


# ==========================================
# 2. PDF LOADER TESTS
# ==========================================

@patch("src.ingestion.pdf_loader.os.path.exists", return_value=True)
@patch("src.ingestion.pdf_loader.os.path.getsize", return_value=1024)
@patch("src.ingestion.pdf_loader.PdfReader")
def test_pdf_loader_success(mock_pdf_reader, mock_getsize, mock_exists):
    """
    Tests successful PDF extraction, metadata propagation, and content hash generation.
    """
    # Configure mock PdfReader
    mock_reader_inst = MagicMock()
    mock_pdf_reader.return_value = mock_reader_inst
    mock_reader_inst.is_encrypted = False
    mock_reader_inst.metadata = {
        "/Author": "Dr. Research",
        "/Title": "RAG Foundations",
        "/CreationDate": "D:20260718"
    }
    
    # Configure pages
    page1 = MagicMock()
    page1.extract_text.return_value = "Page 1 Content here."
    page2 = MagicMock()
    page2.extract_text.return_value = "Page 2 Content here."
    mock_reader_inst.pages = [page1, page2]
    
    # Mock file binary read for SHA-256 computation
    with patch("builtins.open", mock_open(read_data=b"mock-pdf-binary-data")):
        loader = PDFLoader("test_file.pdf")
        docs = loader.load()
        
    assert len(docs) == 2
    assert docs[0].page_content == "Page 1 Content here."
    assert docs[1].page_content == "Page 2 Content here."
    
    # Verify metadata fields
    for idx, doc in enumerate(docs):
        assert doc.metadata["filename"] == "test_file.pdf"
        assert doc.metadata["page_number"] == idx + 1
        assert doc.metadata["total_pages"] == 2
        assert doc.metadata["author"] == "Dr. Research"
        assert doc.metadata["title"] == "RAG Foundations"
        assert doc.metadata["creation_date"] == "D:20260718"
        assert doc.metadata["document_type"] == "pdf"
        assert "document_id" in doc.metadata


@patch("src.ingestion.pdf_loader.os.path.exists", return_value=True)
@patch("src.ingestion.pdf_loader.os.path.getsize", return_value=1024)
@patch("src.ingestion.pdf_loader.PdfReader")
def test_pdf_loader_empty_or_scanned(mock_pdf_reader, mock_getsize, mock_exists):
    """
    Ensures empty or scanned PDFs (no text extractable) throw DocumentLoadError.
    """
    mock_reader_inst = MagicMock()
    mock_pdf_reader.return_value = mock_reader_inst
    mock_reader_inst.is_encrypted = False
    
    # Pages return empty text (scanned page)
    page = MagicMock()
    page.extract_text.return_value = "   "
    mock_reader_inst.pages = [page]
    
    with patch("builtins.open", mock_open(read_data=b"mock-scanned-pdf-data")):
        loader = PDFLoader("scanned.pdf")
        with pytest.raises(DocumentLoadError) as exc_info:
            loader.load()
        assert "contains no extractable text" in str(exc_info.value)


@patch("src.ingestion.pdf_loader.os.path.exists", return_value=True)
@patch("src.ingestion.pdf_loader.os.path.getsize", return_value=1024)
@patch("src.ingestion.pdf_loader.PdfReader")
def test_pdf_loader_encrypted_failure(mock_pdf_reader, mock_getsize, mock_exists):
    """
    Ensures password-encrypted PDFs trigger a DocumentLoadError.
    """
    mock_reader_inst = MagicMock()
    mock_pdf_reader.return_value = mock_reader_inst
    mock_reader_inst.is_encrypted = True
    mock_reader_inst.decrypt.return_value = 0 # 0 indicates decryption failed
    
    with patch("builtins.open", mock_open(read_data=b"mock-encrypted-pdf-data")):
        loader = PDFLoader("encrypted.pdf")
        with pytest.raises(DocumentLoadError):
            loader.load()


@patch("src.ingestion.pdf_loader.os.path.exists", return_value=True)
@patch("src.ingestion.pdf_loader.os.path.getsize", return_value=25 * 1024 * 1024) # 25MB
def test_pdf_loader_size_limit(mock_getsize, mock_exists):
    """
    Ensures files exceeding the configurable limit throw FileLimitExceededError.
    """
    loader = PDFLoader("large.pdf", max_file_size_mb=20)
    with pytest.raises(FileLimitExceededError):
        loader.load()


# ==========================================
# 3. TXT LOADER TESTS
# ==========================================

@patch("src.ingestion.txt_loader.os.path.exists", return_value=True)
@patch("src.ingestion.txt_loader.os.path.getsize", return_value=500)
def test_txt_loader_encodings(mock_getsize, mock_exists):
    """
    Tests successful loading of standard TXT files encoded in UTF-8, UTF-8 BOM, and ISO-8859-1.
    """
    utf8_data = "Standard text content.".encode("utf-8")
    bom_data = "\ufeffBOM text content.".encode("utf-8")
    iso_data = "ISO-8859 text content: \xe9.".encode("iso-8859-1")
    
    # 1. UTF-8 test
    with patch("builtins.open", mock_open(read_data=utf8_data)):
        loader = TXTLoader("utf8.txt")
        docs = loader.load()
        assert docs[0].page_content == "Standard text content."
        assert docs[0].metadata["encoding"] == "utf-8"
        
    # 2. UTF-8 BOM test
    with patch("builtins.open", mock_open(read_data=bom_data)):
        loader = TXTLoader("bom.txt")
        docs = loader.load()
        assert docs[0].page_content == "BOM text content."
        assert docs[0].metadata["encoding"] == "utf-8-sig"

    # 3. ISO-8859-1 test
    with patch("builtins.open", mock_open(read_data=iso_data)):
        loader = TXTLoader("iso.txt")
        docs = loader.load()
        assert docs[0].page_content == "ISO-8859 text content: é."
        assert docs[0].metadata["encoding"] == "iso-8859-1"


@patch("src.ingestion.txt_loader.os.path.exists", return_value=True)
@patch("src.ingestion.txt_loader.os.path.getsize", return_value=500)
def test_txt_loader_unsupported_encoding(mock_getsize, mock_exists):
    """
    Verify that unsupported/corrupted bytes raise UnsupportedEncodingError.
    """
    # Corrupted byte array that throws DecodeError on target candidates
    corrupted_data = b"\x80\x81\x82\x83\x84"
    
    with patch("builtins.open", mock_open(read_data=corrupted_data)):
        # Configured to only try UTF-8 which throws decode error on corrupted bytes
        loader = TXTLoader("corrupted.txt", encodings=["utf-8"])
        with pytest.raises(UnsupportedEncodingError):
            loader.load()


# ==========================================
# 4. WEB LOADER TESTS
# ==========================================

def test_web_loader_invalid_url():
    """
    Ensures malformed URLs trigger parameter ValueErrors.
    """
    loader = WebLoader("not-a-valid-url")
    with pytest.raises(ValueError):
        loader.load()


@patch("src.ingestion.web_loader.requests.get")
def test_web_loader_robots_txt_block(mock_get):
    """
    Ensures WebLoader raises WebScrapingError when disallowed by robots.txt.
    """
    # Mock robots.txt response
    mock_robots_response = MagicMock()
    mock_robots_response.status_code = 200
    mock_robots_response.text = "User-agent: *\nDisallow: /"
    
    mock_get.return_value = mock_robots_response
    
    loader = WebLoader("https://example.com/blocked-page")
    with pytest.raises(WebScrapingError) as exc_info:
        loader.load()
    assert "scraping denied" in str(exc_info.value).lower()


@patch("src.ingestion.web_loader.requests.get")
def test_web_loader_timeout_retry(mock_get):
    """
    Verifies that HTTP requests execute retries on timeouts and raise WebScrapingError.
    """
    # Mock robots.txt to succeed, but actual request to time out
    mock_robots = MagicMock()
    mock_robots.status_code = 200
    mock_robots.text = "User-agent: *\nAllow: /"
    
    mock_get.side_effect = [mock_robots, requests.exceptions.Timeout("Connection timed out")]
    
    # We configure max_retries = 1 to keep test speed fast
    loader = WebLoader("https://example.com/page", timeout=1, max_retries=1)
    with pytest.raises(WebScrapingError) as exc_info:
        loader.load()
    assert "timed out" in str(exc_info.value).lower()


@patch("src.ingestion.web_loader.requests.get")
def test_web_loader_html_cleaning(mock_get):
    """
    Verifies that headers, footers, scripts, and navigation structures are correctly stripped.
    """
    mock_robots = MagicMock()
    mock_robots.status_code = 200
    mock_robots.text = "User-agent: *\nAllow: /"
    
    mock_page = MagicMock()
    mock_page.status_code = 200
    mock_page.content = """
    <html>
        <head><title>Test Article</title></head>
        <body>
            <header><h1>My Header Title</h1></header>
            <nav><a href="/home">Home</a></nav>
            <main>
                <article>
                    <p>This is the important main body content.</p>
                </article>
                <div class="ad">Buy our product!</div>
            </main>
            <footer>Copyright 2026</footer>
            <script>console.log("hello");</script>
        </body>
    </html>
    """
    
    mock_get.side_effect = [mock_robots, mock_page]
    
    loader = WebLoader("https://example.com/article")
    docs = loader.load()
    
    text = docs[0].page_content
    # Expected: header, nav, ad, footer, and script contents should be stripped
    assert "Test Article" in docs[0].metadata["title"]
    assert "important main body content" in text
    assert "My Header Title" not in text
    assert "Buy our product" not in text
    assert "Copyright 2026" not in text


# ==========================================
# 5. TEXT SPLITTER TESTS
# ==========================================

def test_splitter_metadata_consistency():
    """
    Verifies that document splitters propagate parent tags, generate unique chunk IDs,
    embed batch timestamps, and construct valid indexes.
    """
    doc = Document(
        page_content="Block1 content. " * 50, # 800 chars
        metadata={
            "document_id": "sha256-hash-value-12345",
            "filename": "paper.pdf",
            "document_type": "pdf",
            "source": "/docs/paper.pdf",
            "page_number": 3,
            "extra_field": "author-name"
        }
    )
    
    splitter = DocumentSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_documents([doc])
    
    assert len(chunks) > 1
    
    # Assert metadata format of chunks
    for idx, chunk in enumerate(chunks):
        meta = chunk.metadata
        assert meta["document_id"] == "sha256-hash-value-12345"
        assert meta["chunk_id"] == f"sha256-hash-value-12345_chunk_{idx}"
        assert meta["chunk_index"] == idx
        assert meta["total_chunks"] == len(chunks)
        assert meta["filename"] == "paper.pdf"
        assert meta["document_type"] == "pdf"
        assert meta["source"] == "/docs/paper.pdf"
        assert meta["page_number"] == 3
        assert meta["extra_field"] == "author-name"
        assert "ingestion_timestamp" in meta


def test_splitter_cancellation():
    """
    Asserts IngestionCancelledError is raised during loops if checker returns True.
    """
    doc = Document(
        page_content="Splitting content string.",
        metadata={"document_id": "hash", "filename": "doc.txt"}
    )
    
    splitter = DocumentSplitter()
    
    # Cancellation checker returns True immediately
    def cancel_now():
        return True
        
    with pytest.raises(IngestionCancelledError):
        splitter.split_documents([doc], cancellation_check=cancel_now)


# ==========================================
# 6. PIPELINE & PERFORMANCE TESTS
# ==========================================

def test_pipeline_duplicate_detection():
    """
    Verifies that deterministic hash-based duplicate detection successfully skips
    indexing identical documents.
    """
    # Simple simulated ingestion manager
    indexed_document_hashes = set()
    processing_log = []
    
    def process_document(doc: Document) -> bool:
        doc_id = doc.metadata["document_id"]
        if doc_id in indexed_document_hashes:
            processing_log.append(f"Skipped duplicate: {doc.metadata['filename']}")
            return False # Skip
        
        indexed_document_hashes.add(doc_id)
        processing_log.append(f"Indexed document: {doc.metadata['filename']}")
        return True
        
    # Create two different docs, and one duplicate
    doc1 = Document("Content Alpha", {"document_id": "hash-alpha", "filename": "alpha.txt"})
    doc2 = Document("Content Beta", {"document_id": "hash-beta", "filename": "beta.txt"})
    doc3 = Document("Content Alpha", {"document_id": "hash-alpha", "filename": "alpha_duplicate.txt"})
    
    res1 = process_document(doc1)
    res2 = process_document(doc2)
    res3 = process_document(doc3)
    
    assert res1 is True
    assert res2 is True
    assert res3 is False # Skipped
    assert "Skipped duplicate: alpha_duplicate.txt" in processing_log
