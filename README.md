# Smart Research Assistant - RAG Based Knowledge System

A production-quality Retrieval-Augmented Generation (RAG) platform built with a highly decoupled, modular architecture. It is designed to act as an AI Research Assistant capable of answering research questions grounded in source documentation like research papers, policies, technical manuals, uploaded PDFs, TXT files, and Web pages.

The code sits strictly segregated into distinct layers, allowing simple, config-driven swaps for Vector Stores, LLM Providers, and Ingestion models.

---

## 🌟 Key Architecture & Highlights

- **Python Version**: Recommended Python `3.11`.
- **UI Orchestration**: Built entirely using Streamlit (`app.py` acts strictly as the visual orchestrator; business logic is kept in the `src/` module).
- **LLM Provider Abstraction**: Supports OpenAI, Anthropic, Ollama (Local), and Hugging Face remote endpoints via a unified factory interface (`BaseLLM`).
- **Embedding Provider**: Hugging Face Sentence Transformers as the default.
- **Unified Vector Database Wrapper**: Supports switching between `ChromaDB` and `FAISS` using the same client contract (`BaseVectorStore`).
- **Rich User Interface**:
  - Top-level KPI metrics dashboard cards.
  - Interactive QA Chat interface with stream response support.
  - Interactive source citation badges.
  - Collapsible retrieved context inspector/viewer.
  - RAG pipeline quality validation metrics evaluation panel.
  - Ingestion progress indicators.
  - Dynamic sidebar selectors.

---

## 📂 Folder Layout

```text
smart-research-assistant-rag/
├── .streamlit/
│   └── config.toml             # Streamlit design parameters (theme, port, etc.)
├── app.py                      # UI Orchestration layer (only UI state & layout)
├── requirements.txt            # Mutually compatible stable versions (LangChain v0.2, etc.)
├── README.md                   # System overview, installation, and usage instructions
├── .gitignore                  # Git exclusions (credentials, caches, DB files)
├── .env.example                # Configurable template with keys & settings
├── assets/
│   ├── logo.png                # App Logo
│   └── styles.css              # Custom styles (cards, chat bubbles, sidebar highlights)
├── docs/
│   ├── architecture.md         # Architecture diagrams & walkthroughs
│   ├── workflow.md             # RAG ingestion/query workflow diagrams
│   └── components.md           # Description of modular blocks
├── scripts/
│   ├── deploy_check.py         # Deployment readiness validator
│   └── generate_logo.py        # Programmatic logo builder
├── src/
│   ├── __init__.py
│   ├── config.py               # Env var loading and validation (Pydantic settings)
│   ├── constants.py            # System-wide static variables
│   ├── logger.py               # Configured logging framework
│   ├── settings.py             # Streamlit page layout configurations
│   ├── prompts.py              # Prompt templates
│   ├── ingestion/              # Loaders (PDF, TXT, Web) & Splitters
│   ├── embeddings/             # Embedding models factory wrappers
│   ├── vectorstores/           # Chroma & FAISS wrapper integrations
│   ├── retrieval/              # Query routing and Search execution
│   ├── llm/                    # OpenAI, Anthropic, Ollama, Hugging Face engines
│   ├── evaluation/             # Pipeline validation (Faithfulness, Relevance)
│   └── ui/                     # Reusable components & styled layers
└── tests/                      # Pytest unit tests suite
```

---

## 🛠️ Getting Started & Setup

### 1. Prerequisites
- Python 3.11+
- Virtual environment tool (`venv` or `conda`)

### 2. Installation
Clone the repository and create/activate your virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variable Configuration
Duplicate the example environment file and fill in your credentials:

```bash
copy .env.example .env
```

Open `.env` and fill in your API tokens (e.g., `OPENAI_API_KEY`, `LOG_LEVEL`, etc.).

---

## 🧪 Verification & Testing

### 1. Run Pre-flight Deployment Check
Run the pre-flight script to verify Python versions, folder structure, configuration files, and package imports:

```bash
python scripts/deploy_check.py
```

### 2. Run Automated Pytest Suite
Run the automated test cases confirming config validation, document splitting, multi-provider model loading, and database management:

```bash
pytest tests/
```

### 3. Run Streamlit UI Dashboard
Launch the dashboard locally:

```bash
streamlit run app.py
```
*Note: If no API keys are configured, the application automatically runs using local Mock modules, allowing full navigation and dashboard testing without hitting endpoint rate limits.*