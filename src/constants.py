"""
This module defines the project-wide constants for the Smart Research Assistant application.
"""

import os
from typing import List, Set

# Project Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
CSS_PATH = os.path.join(ASSETS_DIR, "styles.css")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
DEFAULT_LOG_PATH = os.path.join(BASE_DIR, "logs", "app.log")

# Document Ingestion Settings
SUPPORTED_EXTENSIONS: Set[str] = {".pdf", ".txt", ".html", ".htm"}
DEFAULT_CHUNK_SIZE: int = 1000
DEFAULT_CHUNK_OVERLAP: int = 200
DEFAULT_MAX_UPLOAD_SIZE_MB: int = 20

# Embedding Provider Configuration
DEFAULT_EMBEDDING_PROVIDER: str = "huggingface"
DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# Vector Store Configuration
DEFAULT_VECTOR_STORE_TYPE: str = "chroma"
DEFAULT_VECTOR_STORE_DIR: str = os.path.join(BASE_DIR, "data", "vectorstore")

# LLM Providers Configuration
DEFAULT_LLM_PROVIDER: str = "openai"
DEFAULT_LLM_MODEL: str = "gpt-4o-mini"
OLLAMA_FALLBACK_URL: str = "http://localhost:11434"

# Supported Providers List
SUPPORTED_LLM_PROVIDERS: List[str] = ["openai", "anthropic", "ollama", "huggingface"]
SUPPORTED_EMBEDDING_PROVIDERS: List[str] = ["huggingface", "openai"]
SUPPORTED_VECTOR_STORES: List[str] = ["chroma", "faiss"]

# Streamlit Page Titles
APP_TITLE: str = "Smart Research Assistant"
APP_SUBTITLE: str = "RAG Based Knowledge System"
APP_ICON: str = "🔍"

# Logs configuration
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
