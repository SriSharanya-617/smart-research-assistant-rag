# User Guide - Smart Research Assistant 📖
> Complete handbook detailing how to manage documents, configure search parameters, inspect retrieval diagnostics, and run evaluations.

🌐 **Live Deployed App:** [smart-research19.streamlit.app](https://smart-research19.streamlit.app/)

---

## 📖 Table of Contents
1. [Interface Navigation Overview](#interface-navigation-overview)
2. [Document Ingestion Management](#document-ingestion-management)
3. [RAG Strategy Tuning Settings](#rag-strategy-tuning-settings)
4. [Interactive Chatroom Console](#interactive-chatroom-console)
5. [Diagnostics & Retrieval Inspector](#diagnostics--retrieval-inspector)
6. [Evaluation & Benchmarking Tab](#evaluation--benchmarking-tab)

---

## 1. Interface Navigation Overview
Upon starting the app (`streamlit run app.py`), the UI displays a clean split layout:
- **Left Sidebar**: Configuration controls, upload fields, and database reset triggers.
- **Main Area**: Divided into two tabs:
  - **💬 Q&A Chatroom**: Interactive chatbot window on the left, document list and diagnostic inspector panel on the right.
  - **📊 RAG Benchmark Evaluations**: Bulk dataset testing console, latency line plots, quality stats, and report exporter download buttons.

---

## 2. Document Ingestion Management
To index context sources into the local vector database:

1. **File Uploads**: Drag and drop PDF or TXT files onto the uploader in the sidebar.
2. **Website URL Scraping**: Type or paste a URL (e.g. `https://example.com/article`) into the web scraper text box.
3. **Index Execution**: Click the **⚡ Process & Index** button. An active progress bar shows parsing, chunk splitting, embedding generation, and database indexing.
4. **Inspect Documents**: In the right-hand panel of the Chatroom tab, view the loaded filenames list showing total chunks generated. Click the trash icon (`🗑️`) next to any filename to remove its vector chunks from the database index.

### Screenshot Placeholders
- **Managed Index Documents Panel**: `assets/screenshots/documents_panel.png`
- **Ingestion Sidebar Controls**: `assets/screenshots/upload_documents.png`

---

## 3. RAG Strategy Tuning Settings
All API keys and provider models are loaded directly from a `.env` file in your project root. The UI sidebar has no API key fields to prevent credential exposure.

### Environment variables configuration:
1. Create a `.env` file from the provided `.env.example` in the project root:
   ```bash
   cp .env.example .env
   ```
2. Configure credentials and model choices in the `.env` file:
   ```env
   GOOGLE_API_KEY=your_gemini_key
   OPENAI_API_KEY=your_openai_key
   LLM_PROVIDER=gemini
   LLM_MODEL=gemini-1.5-flash
   VECTOR_STORE_TYPE=faiss
   EMBEDDING_PROVIDER=huggingface
   EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   TOP_K=4
   TEMPERATURE=0.2
   ```

At runtime, you can still alter retrieval parameters and search boundaries using the sidebar selectors:

- **Vector Database**: Choose `chroma` (persistent SQLite files) or `faiss` (local memory index).
- **Embeddings Model**: Default is `BAAI/bge-small-en-v1.5` (384 dimensions, high MTEB retrieval rank).
- **LLM Provider**: Choose `Gemini` (default model `gemini-1.5-flash`), `OpenAI` (`gpt-4o-mini`), `Ollama` (local hosting), or `Mock` (offline testing).
- **Chunk Parameters**: Adjust Chunk Size and Overlap to refine search granularity.
- **Retrieval Parameters**: Choose `semantic` (Cosine/L2 similarity distance) or `mmr` (Maximal Marginal Relevance for diverse chunk selection). Set Top-K to limit retrieved chunks.

---

## 4. Interactive Chatroom Console
Ask questions about the indexed documents:
- Type your prompt in the chat input bar.
- If context is found, the assistant responds with retrieved facts, followed by formatted source citations showing file names and page numbers.
- If the database is empty, the assistant returns a configurable safe fallback message immediately, avoiding unnecessary LLM calls.

### Screenshot Placeholders
- **QA Chat Room Bubble Dialogue**: `assets/screenshots/chat_room.png`

---

## 5. Diagnostics & Retrieval Inspector
Located on the right-hand column of the Q&A tab:
- **Original & Processed Query**: Shows the sanitizations performed (Unicode normalization, whitespace consolidation).
- **Query Complexity (Intent)**: Shows the detected intent (e.g. `comparison request`, `summary request`, `definition request`).
- **Cache Hit**: Shows if the query was served from the TTL cache.
- **Retrieved Chunk Extracts**: Expand to inspect the exact text snippets retrieved, their distance/similarity scores, and the selection explanation logic.

### Screenshot Placeholders
- **Retrieval Inspector Diagnostic Expanders**: `assets/screenshots/inspector_console.png`

---

## 6. Evaluation & Benchmarking Tab
To run a batch validation test:

1. **Upload Dataset**: Upload a CSV or JSON file containing `question` and optional `expected_answer` columns. Alternatively, type questions directly in the manual text area (one per line).
2. **Execute Benchmark**: Click **⚡ Run Evaluation Benchmark**.
3. **Analyze Results**: View metric cards summarizing Retrieval Success Rates, E2E Latencies, and Heuristic Overlap indicators (Faithfulness, Relevance, Precision).
4. **Visualize Charts**: Inspect line plots of latency trends, bar charts of confidence classes, and histograms of similarity scores.
5. **Download Reports**: Click to download **Markdown**, **CSV**, **JSON**, or **HTML Dashboard** reports.
6. **History Logs**: Inspect the historical logs table showing previous runs.

### Screenshot Placeholders
- **Benchmarking Visualizations Page**: `assets/screenshots/evaluation_console.png`
