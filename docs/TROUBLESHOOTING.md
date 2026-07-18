# Troubleshooting Guide - Smart Research Assistant 🛠️
> Solutions for dependency conflicts, API rate limits, missing keys, Streamlit startup failures, vector database errors, and model download issues.

---

## 📖 Table of Contents
1. [Dependency Conflicts](#dependency-conflicts)
2. [API Authorization & Rate Limits](#api-authorization--rate-limits)
3. [Streamlit Startup Failures](#streamlit-startup-failures)
4. [Vector Database & Indexing Errors](#vector-database--indexing-errors)
5. [Embedding Model Download Failures](#embedding-model-download-failures)

---

## 1. Dependency Conflicts

### Problem: `ModuleNotFoundError: No module named 'pydantic_settings'`
This error occurs when you run Streamlit in an environment where the dependencies are not installed.
- **Solution**: Install the required packages in your active Python environment:
  ```bash
  pip install -r requirements.txt
  ```

### Problem: `ImportError: cannot import name 'Chroma' from 'langchain_community.vectorstores'`
This happens when you have an older version of `langchain_community` installed.
- **Solution**: Upgrade the package:
  ```bash
  pip install --upgrade langchain-community
  ```

---

## 2. API Authorization & Rate Limits

### Problem: `APIKeyError: OpenAI API Key is missing` or `Gemini API Key is missing`
You did not set the authorization key in the environment or in the UI.
- **Solution**:
  - Add `GOOGLE_API_KEY=your_key` or `OPENAI_API_KEY=your_key` to a `.env` file in the root folder.
  - Or paste the key into the password input box in the Streamlit sidebar.

### Problem: `ProviderUnavailableError: Rate Limit Exceeded (HTTP 429)`
This occurs when you exceed your LLM API provider's tier limits (often on free tiers of Gemini or OpenAI).
- **Solution**:
  - Reduce the benchmark dataset size to prevent rapid API requests.
  - Add slight delay gaps between queries in `Runner`.
  - Or switch the LLM Provider to `mock` in the sidebar to test offline.

---

## 3. Streamlit Startup Failures

### Problem: Streamlit app runs but closes instantly in the console
This usually indicates a Python version conflict or an active background process locking port `8501`.
- **Solution**:
  - Force kill any active Streamlit processes on Windows:
    ```powershell
    Stop-Process -Name streamlit -Force
    ```
  - Force kill on Linux/macOS:
    ```bash
    kill -9 $(lsof -t -i:8501)
    ```
  - Run on a custom port:
    ```bash
    streamlit run app.py --server.port=8502
    ```

---

## 4. Vector Database & Indexing Errors

### Problem: `DimensionMismatchError: Embedding vector dimension does not match index dimension`
You attempted to add documents to an existing index using a different embedding model than the one used to initialize it (e.g. indexing with `all-MiniLM-L6-v2` into an index created with `bge-small-en-v1.5`).
- **Solution**: Click the **🗑️ Reset Database Index** button in the sidebar to clear the vector index files, select your new model, and re-index the documents.

### Problem: `CorruptedIndexError: Could not load or deserialize FAISS database`
This occurs when local FAISS index files (`index.faiss`, `index.pkl`) are corrupted or have been overwritten with invalid bytes.
- **Solution**: Delete the folder `data/vectorstore` (or click **Reset Database Index** in the UI) to let the system generate a fresh, clean index.

---

## 5. Embedding Model Download Failures

### Problem: Model download hangs or raises connection errors when initializing `sentence-transformers`
This happens when you have slow internet connections or firewalls block Hugging Face Hub downloads (`huggingface.co`).
- **Solution**:
  - Ensure your system is connected to the internet.
  - Configure a custom local cache folder in `.env`:
    ```env
    EMBEDDING_CACHE_DIR=./data/models_cache
    ```
  - If you are completely offline, select `mock` as the embedding provider in the sidebar to run the system with dummy vectors.
