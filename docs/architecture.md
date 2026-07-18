# System Architecture

The **Smart Research Assistant** is built on a highly modular, decoupled architecture where the user interface, business logic, ingestion pipelines, storage interfaces, and models are separated into distinct modules. 

This decoupling ensures that:
- Adding a new file loader format does not affect the UI or LLM logic.
- Switching from FAISS to ChromaDB (or vice versa) is a configuration change.
- Swapping the LLM provider (e.g., OpenAI to Anthropic or a local Ollama model) does not break prompt formatting.
- Evaluation runs as an independent post-processing/validation layer.

---

## Component Architecture

Below is the component relationship diagram illustrating the data flow and orchestration.

```mermaid
graph TD
    subgraph UI [Streamlit UI Orchestration Layer]
        App[app.py]
        UIComp[src/ui/components.py]
        Styles[src/ui/styles.py]
    end

    subgraph Config [Configuration & Utilities]
        Settings[src/settings.py]
        Conf[src/config.py]
        Const[src/constants.py]
        Log[src/logger.py]
        Utils[src/utils/helpers.py]
    end

    subgraph Ingestion [Ingestion Pipeline]
        BaseLoad[src/ingestion/base.py]
        Loaders[PDFLoader / TXTLoader / WebLoader]
        Splitter[src/ingestion/splitter.py]
    end

    subgraph Storage [Vector Stores]
        VSBase[src/vectorstores/base.py]
        Chroma[src/vectorstores/chroma_store.py]
        FAISS[src/vectorstores/faiss_store.py]
        VSMan[src/vectorstores/manager.py]
    end

    subgraph Models [Models & Abstractions]
        EmbedBase[src/embeddings/base.py]
        EmbedFact[src/embeddings/factory.py]
        LLMBase[src/llm/base.py]
        LLMFact[src/llm/factory.py]
    end

    subgraph Eval [Evaluation Module]
        Evaluator[src/evaluation/evaluator.py]
    end

    %% UI Connections
    App --> UIComp
    App --> Styles
    App --> Conf
    App --> Settings

    %% Core Data flow
    App --> Ingestion
    App --> Storage
    App --> Models
    App --> Eval

    %% Modular Ingestion
    BaseLoad --> Loaders
    Loaders --> Splitter

    %% Storage & Embeddings
    VSMan --> VSBase
    VSBase --> Chroma
    VSBase --> FAISS
    Chroma -.-> EmbedFact
    FAISS -.-> EmbedFact

    %% Models
    EmbedFact --> EmbedBase
    LLMFact --> LLMBase
```

## Architectural Highlights

1. **Streamlit App (`app.py`)**: Responsible only for user session states, routing tabs (Upload, Chat, Evaluation), and layout rendering.
2. **Configuration Class (`src/config.py`)**: Uses Pydantic Settings (or a structured validation class) to read and assert environment variables from `.env`.
3. **Common Interfaces**:
   - `BaseDocumentLoader`: Guarantees consistent format retrieval.
   - `BaseVectorStore`: Abstracts vector operations so that swapping from ChromaDB to FAISS requires only changing `VECTOR_STORE_TYPE` in `.env`.
   - `BaseLLMProvider`: Standardizes request/response schemas across OpenAI, Anthropic, Ollama, and Hugging Face.
