# RAG Workflows

The **Smart Research Assistant** runs two core workflows:
1. **Document Ingestion Workflow**: Parsing uploaded files, splitting them into logical chunks, converting chunks to embeddings, and loading them into the configured Vector Store.
2. **Query & Retrieval Workflow**: Transforming user questions, searching the Vector Store, assembling context-augmented prompts, executing LLM inference, and evaluating quality.

---

## 1. Document Ingestion Workflow

This workflow is triggered when a user uploads documents via the UI dashboard.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as app.py (Streamlit)
    participant LManager as Ingestion Loaders
    participant Splitter as Text Splitter
    participant Embed as Embedding Factory
    participant DB as VectorStore Manager

    User->>UI: Upload Files (PDF / TXT) or Web URL
    UI->>LManager: Route file to specific loader
    Note over LManager: PDFLoader, TXTLoader, or WebLoader parses text
    LManager->>UI: Return List[Document]
    UI->>Splitter: Split raw documents into chunks
    Splitter->>UI: Return List[Chunked Documents]
    UI->>Embed: Convert text chunks to float vectors
    Embed->>UI: Return Embedded Vectors
    UI->>DB: Upsert vectors & metadata to ChromaDB/FAISS
    DB-->>User: Update UI Progress Bar (Success Toast)
```

---

## 2. Query & Retrieval Workflow

This workflow is executed when a user submits a search or question in the chat interface.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as app.py (Streamlit)
    participant VS as VectorStore Interface
    participant Prompt as Prompts Module
    participant LLM as LLM Provider Factory
    participant Eval as Evaluator

    User->>UI: Input Question / Query
    UI->>VS: Query similarity search (Chroma / FAISS)
    VS->>UI: Return top-K relevant text chunks + citations
    UI->>Prompt: Format context and question into prompt template
    Prompt->>UI: Return formatted prompt string
    UI->>LLM: Send prompt to LLM (OpenAI / Anthropic / Ollama / HF)
    LLM->>UI: Stream/Return model response
    UI->>Eval: Calculate metrics (Faithfulness & Relevance)
    Eval->>UI: Return Evaluation Scores
    UI-->>User: Display response, source citations, context, and quality metrics
```
