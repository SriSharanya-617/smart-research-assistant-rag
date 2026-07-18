# Component Descriptions

This document outlines the detailed responsibilities, structures, and configuration hooks for each directory and module in `src/`.

---

## 1. Configurations & Constants

### `src/config.py`
- **Purpose**: Dynamically loads, validates, and casts configurations from environment variables.
- **Key Class**: `AppConfig` (extending Pydantic `BaseSettings`).
- **Validations**: Asserts that if `LLM_PROVIDER` is `openai`, then `OPENAI_API_KEY` is present. Checks standard folder directories.

### `src/constants.py`
- **Purpose**: Holds hardcoded settings, fallback values, message strings, and static system settings.
- **Key Constants**: Max file size limits, standard layout colors, supported loader extensions, default system prompts.

### `src/logger.py`
- **Purpose**: Implements a standard Python logging wrapper with rotating file logging to `logs/app.log` and formatted CLI console logs.

### `src/settings.py`
- **Purpose**: Orchestrates Streamlit page layouts, setups browser titles, page icon assets, and responsive wide screens.

---

## 2. Ingestion Module (`src/ingestion/`)

- **`base.py`**: Defines abstract class `BaseDocumentLoader` requiring a `load(self) -> List[Document]` implementation.
- **`pdf_loader.py`**: Utilizes `pypdf` to parse PDFs, extracting page contents and page metadata.
- **`txt_loader.py`**: Reads standard text files, preserving formatting and offsets.
- **`web_loader.py`**: Integrates `requests` and `beautifulsoup4` to fetch, strip HTML wrappers, and load web content.
- **`splitter.py`**: Wraps text splitting utilities to slice documents into custom chunks while keeping metadata mappings.

---

## 3. Embeddings Module (`src/embeddings/`)

- **`base.py`**: Common interface `BaseEmbeddings` declaring `embed_documents()` and `embed_query()`.
- **`factory.py`**: Spawns correct embedder instance (e.g. Hugging Face Sentence Transformers or OpenAI) depending on the configuration.

---

## 4. Vector Store Module (`src/vectorstores/`)

- **`base.py`**: Unified storage interface `BaseVectorStore` exposing `add_documents()`, `similarity_search()`, and `delete()`.
- **`chroma_store.py`**: Integration wrapper for ChromaDB.
- **`faiss_store.py`**: Integration wrapper for FAISS.
- **`manager.py`**: Dynamic store selector to load, initialize, or clear index databases dynamically.

---

## 5. Models Module (`src/llm/`)

- **`base.py`**: Interface `BaseLLM` with unified signature for synchronous and streaming model calls.
- **`factory.py`**: Instantiates and connects ChatOpenAI, ChatAnthropic, ChatOllama, or Hugging Face endpoint classes.

---

## 6. UI Module (`src/ui/`)

- **`components.py`**: Holds modular UI rendering methods:
  - `render_dashboard_cards()`: Shows metric tiles.
  - `render_chat_interface()`: Standard input boxes and chat bubble containers.
  - `render_citations()`: Display sources nicely.
  - `render_context_viewer()`: Visualizes retrieved chunks.
  - `render_evaluation_panel()`: Display alignment metrics.
  - `render_sidebar_controls()`: Dropdowns and upload widgets.
- **`styles.py`**: Loads and injects `assets/styles.css`.

---

## 7. Evaluation & Utilities

- **`src/evaluation/evaluator.py`**: Placeholders for measuring response similarity, context faithfulness, and hallucinations.
- **`src/utils/helpers.py`**: Format utilities, token counters, and size converter calculations.
