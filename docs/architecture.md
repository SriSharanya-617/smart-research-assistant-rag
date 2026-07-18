# System Architecture - Smart Research Assistant 📐
> High-level design, decoupled package dependencies, metadata propagation paths, and deployment specifications.

---

## 📖 Table of Contents
1. [Overall System Component Architecture](#overall-system-component-architecture)
2. [Class Interaction Diagram (High Level)](#class-interaction-diagram-high-level)
3. [Sequence Diagram for Question Answering](#sequence-diagram-for-question-answering)
4. [Deployment Architecture Diagram](#deployment-architecture-diagram)
5. [Decoupled Architecture & SOLID Principles](#decoupled-architecture--solid-principles)

---

## 1. Overall System Component Architecture
The application is structured into isolated packages under `src/`. No backend package imports from Streamlit, maintaining strict separation of concerns:

```mermaid
graph TD
    UI[Streamlit UI App: app.py] -->|1. Triggers Ingestion| Ingestion[src.ingestion]
    UI -->|2. Queries pipeline| Retrieval[src.retrieval]
    UI -->|3. Runs benchmark| Evaluation[src.evaluation]
    
    Ingestion -->|Extracts Document list| VectorStore[src.vectorstores]
    Retrieval -->|Queries Vector database| VectorStore
    Retrieval -->|Embeds query text| Embeddings[src.embeddings]
    
    VectorStore -->|Generates embeddings| Embeddings
    
    RAG[RAGPipeline: src.llm] -->|Coordinates| Retrieval
    RAG -->|Queries completions| LLM[src.llm providers]
    
    UI -->|Coordinates chat & answers| RAG
```

---

## 2. Class Interaction Diagram (High Level)
Shows class relations and factories:

```mermaid
classDiagram
    class BaseDocumentLoader {
        +load() List~Document~
    }
    class PDFLoader {
        +load() List~Document~
    }
    class TXTLoader {
        +load() List~Document~
    }
    class WebLoader {
        +load() List~Document~
    }
    
    BaseDocumentLoader <|-- PDFLoader
    BaseDocumentLoader <|-- TXTLoader
    BaseDocumentLoader <|-- WebLoader

    class BaseEmbeddingProvider {
        +embed_documents(texts)
        +embed_query(text)
        +get_dimension()
    }
    class SentenceTransformerProvider {
        +embed_documents(texts)
    }
    BaseEmbeddingProvider <|-- SentenceTransformerProvider
    
    class EmbeddingFactory {
        +get_embeddings(provider, model_name) BaseEmbeddingProvider
    }
    EmbeddingFactory --> BaseEmbeddingProvider : instantiates

    class BaseVectorStore {
        +add_documents(documents)
        +similarity_search_with_score(query)
        +max_marginal_relevance_search(query)
    }
    class ChromaVectorStore {
        +add_documents(documents)
    }
    class FAISSVectorStore {
        +add_documents(documents)
    }
    BaseVectorStore <|-- ChromaVectorStore
    BaseVectorStore <|-- FAISSVectorStore

    class VectorStoreFactory {
        +get_vector_store(store_type, embeddings) BaseVectorStore
    }
    VectorStoreFactory --> BaseVectorStore : instantiates
```

---

## 3. Sequence Diagram for Question Answering
The execution trace of a query processed by `RAGPipeline`:

```mermaid
sequenceDiagram
    autonumber
    actor User as User Interface (app.py)
    participant RAG as RAGPipeline (src.llm)
    participant Retrieval as RetrievalPipeline (src.retrieval)
    participant Store as VectorStore (src.vectorstores)
    participant LLM as BaseLLMProvider (src.llm)
    participant Evaluator as EvaluationEngine (src.evaluation)

    User->>RAG: answer_question("What is RAG?")
    RAG->>Retrieval: retrieve("What is RAG?")
    Retrieval->>Retrieval: Preprocess query & Check cache
    
    alt Cache Miss
        Retrieval->>Store: similarity_search_with_score()
        Store-->>Retrieval: return matching document chunks + scores
    end
    
    Retrieval->>Retrieval: Estimate confidence & Annotate metadata
    Retrieval-->>RAG: return retrieved_docs
    
    RAG->>RAG: Build prompt template
    RAG->>LLM: generate(formatted_prompt)
    LLM-->>RAG: return raw answer text
    
    RAG->>RAG: Deduplicate & format citations
    RAG->>Evaluator: evaluate_response()
    Evaluator-->>RAG: return Faithfulness & Relevance proxy scores
    
    RAG-->>User: return final_answer + citations + diagnostic report
```

---

## 4. Deployment Architecture Diagram
Shows the physical packaging and networking configurations:

```mermaid
graph LR
    UserApp[User Browser] -->|HTTP / WebSockets| StreamlitCloud[Streamlit Community Cloud Container]
    
    subgraph Container Environment
        StreamlitCloud -->|Reads| LocalFAISS[Local FAISS Index Files]
        StreamlitCloud -->|Executes CPU inference| LocalHF[Local PyTorch Sentence-Transformers model]
    end
    
    StreamlitCloud -->|API Request| GoogleGemini[Google Gemini API]
    StreamlitCloud -->|API Request| OpenAI[OpenAI API]
```

---

## 5. Decoupled Architecture & SOLID Principles

- **Single Responsibility (SRP)**: The text preprocessing module does nothing except sanitize text blocks; it has no awareness of FAISS or Chroma. The embedding factory deals only with model caching, not text chunking.
- **Open/Closed (OCP)**: Adding a new LLM provider (such as Anthropic or Cohere) or a new vector database (such as Qdrant or Pinecone) requires writing a new subclass without modifying any of the existing pipeline code.
- **Liskov Substitution (LSP)**: `ChromaVectorStore` and `FAISSVectorStore` are fully interchangeable. The application behaves identically regardless of which database is swapped in.
- **Interface Segregation (ISP)**: Custom loaders conform to `BaseDocumentLoader`, exposing only `.load()` to the preprocessing parser.
- **Dependency Inversion (DIP)**: High-level modules (like the RAG Pipeline orchestrator) depend only on abstractions (`BaseVectorStore`, `BaseLLMProvider`, `BaseEmbeddingProvider`) rather than concrete library packages.
