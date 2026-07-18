# Installation Guide - Smart Research Assistant ⚙️
> Step-by-step setup guides for Windows, Linux, macOS, API configurations, and deployment procedures.

---

## 📖 Table of Contents
1. [Pre-requisites](#pre-requisites)
2. [Windows Setup](#windows-setup)
3. [Linux Setup](#linux-setup)
4. [macOS Setup](#macos-setup)
5. [API Keys & Environment Configuration](#api-keys--environment-configuration)
6. [Launching the Streamlit UI](#launching-the-streamlit-ui)
7. [Running the Test Suite](#running-the-test-suite)
8. [Docker Deployment](#docker-deployment)

---

## 1. Pre-requisites
- **Python**: `Python 3.11` or `Python 3.12` installed.
- **Git**: Installed for cloning the repository.
- **Hardware**: CPU with at least 8GB RAM (16GB recommended). CUDA-compatible GPU is optional but supported.

---

## 2. Windows Setup
Perform these commands in **PowerShell** or **Command Prompt** (run as Administrator if needed):

1. **Clone the Repository**:
   ```powershell
   git clone https://github.com/SriSharanya-617/smart-research-assistant-rag.git
   cd smart-research-assistant-rag
   ```
2. **Create a Virtual Environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. **Install Dependencies**:
   *Note: If you run into compiler errors for C++ packages when installing FAISS, make sure "Desktop development with C++" is checked in your Visual Studio Build Tools installer.*
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 3. Linux Setup
Perform these commands in your bash terminal (tested on Ubuntu 22.04/24.04):

1. **Install Python Development Headers**:
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv python3-dev build-essential -y
   ```
2. **Clone and Navigate**:
   ```bash
   git clone https://github.com/SriSharanya-617/smart-research-assistant-rag.git
   cd smart-research-assistant-rag
   ```
3. **Activate Environment and Install**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 4. macOS Setup
Perform these commands in your Terminal app (compatible with Intel and Apple Silicon M1/M2/M3 chips):

1. **Install Homebrew (if not present) and Python**:
   ```bash
   # Install Xcode Command Line Tools
   xcode-select --install
   brew install python@3.11 git
   ```
2. **Clone and Configure**:
   ```bash
   git clone https://github.com/SriSharanya-617/smart-research-assistant-rag.git
   cd smart-research-assistant-rag
   ```
3. **Setup and Install**:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 5. API Keys & Environment Configuration
API keys and variables are loaded exclusively from a `.env` file in the project root. The UI does not query for keys.

### Setup Instructions:
1. Duplicate the template environment file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and fill in your API authorization keys:
   - **Google Gemini**: Obtain key from [Google AI Studio](https://aistudio.google.com/) and assign to `GOOGLE_API_KEY`.
   - **OpenAI**: Obtain key from [OpenAI API console](https://platform.openai.com/) and assign to `OPENAI_API_KEY`.
3. Configure model settings and parameters:
   - `LLM_PROVIDER`: Choose your provider (`gemini`, `openai`, `ollama`, or `mock`).
   - `LLM_MODEL`: Choose model name (e.g. `gemini-1.5-flash` or `gpt-4o-mini`).
   - `VECTOR_STORE_TYPE`: Select vector database (`faiss` or `chroma`).

### Example `.env` Configuration:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
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

---

## 6. Launching the Streamlit UI
Ensure your virtual environment is active, then run:
```bash
streamlit run app.py
```
This automatically opens your default browser to `http://localhost:8501`.

---

## 7. Running the Test Suite
Ensure the test suite compiles and runs correctly:
```bash
python -m pytest tests/ -v
```

---

## 8. Docker Deployment
A basic Docker configuration allows containerizing the RAG app:

1. **Create `Dockerfile` in the root folder**:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   EXPOSE 8501
   ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```
2. **Build and Run Docker Container**:
   ```bash
   docker build -t smart-research-rag .
   docker run -p 8501:8501 --env-file .env smart-research-rag
   ```
