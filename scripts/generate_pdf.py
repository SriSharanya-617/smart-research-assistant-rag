import os

def make_minimal_pdf(filename, text_lines):
    # Standard minimal PDF objects mapping
    objs = []
    
    # 1. Catalog
    objs.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    # 2. Pages
    objs.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    # 3. Page
    objs.append(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n")
    # 4. Font
    objs.append(b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    
    # Stream content formatting for text lines
    stream_content = b"BT\n/F1 12 Tf\n"
    y = 700
    for line in text_lines:
        stream_content += f"50 {y} Td\n({line}) Tj\n".encode('latin1')
        # Reset relative x coord for next line
        stream_content += b"-50 0 Td\n"
        y = -20  # relative offset for next line
        
    stream_content += b"ET\n"
    
    # 5. Contents Stream Object
    objs.append(
        f"5 0 obj\n<< /Length {len(stream_content)} >>\nstream\n".encode('latin1')
        + stream_content
        + b"\nendstream\nendobj\n"
    )
    
    final_pdf = b"%PDF-1.4\n"
    offsets = {}
    
    for i, obj_bytes in enumerate(objs):
        obj_num = i + 1
        offsets[obj_num] = len(final_pdf)
        final_pdf += obj_bytes
        
    startxref = len(final_pdf)
    
    # Generate xref table
    xref = f"xref\n0 {len(objs) + 1}\n0000000000 65535 f\n".encode('latin1')
    for i in range(len(objs)):
        obj_num = i + 1
        xref += f"{offsets[obj_num]:010d} 00000 n\n".encode('latin1')
        
    trailer = (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
        f"startxref\n{startxref}\n%%EOF\n"
    ).encode('latin1')
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        f.write(final_pdf + xref + trailer)

if __name__ == "__main__":
    lines = [
        "Retrieval-Augmented Generation (RAG) Overview",
        "A guide to building production-grade RAG knowledge systems.",
        "",
        "Retrieval-Augmented Generation (RAG) is a technique that optimizes the output of a large",
        "language model, so it references an authoritative knowledge base outside of its training data",
        "sources before generating a response.",
        "",
        "This document serves as a sample PDF file to test the ingestion pipeline of the Smart",
        "Research Assistant. The loader should parse this document, split it into chunks, generate",
        "vector embeddings using Sentence Transformers, and index them into the local vector store.",
        "",
        "Key concepts of RAG include:",
        "1. Document Ingestion: Loading and splitting PDF, TXT, and Web documents.",
        "2. Embedding Generation: Converting text segments into dense vector representations.",
        "3. Vector Database Indexing: Storing and indexing vectors in ChromaDB or FAISS.",
        "4. Similarity Search: Fetching the most relevant text chunks matching a user query.",
        "5. Prompt Augmentation: Structuring prompts with context chunks for LLM completions.",
    ]
    make_minimal_pdf("sample_data/sample.pdf", lines)
    print("Successfully generated sample_data/sample.pdf")
