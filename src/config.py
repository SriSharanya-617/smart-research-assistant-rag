"""
Configuration loader utilizing Pydantic Settings.
Loads keys from environment variables and `.env` files with validation constraints.
"""

import os
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from src.constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_MAX_UPLOAD_SIZE_MB,
    DEFAULT_VECTOR_STORE_DIR,
    DEFAULT_VECTOR_STORE_TYPE,
    SUPPORTED_EMBEDDING_PROVIDERS,
    SUPPORTED_LLM_PROVIDERS,
    SUPPORTED_VECTOR_STORES,
)

class AppConfig(BaseSettings):
    """
    Application Configuration schema.
    Tries loading variables from standard OS environment and looks for '.env' file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # General Settings
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FILE: str = Field(default="logs/app.log")

    # LLM Settings
    LLM_PROVIDER: str = Field(default=DEFAULT_LLM_PROVIDER)
    LLM_MODEL: str = Field(default=DEFAULT_LLM_MODEL)
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None)
    HUGGINGFACEHUB_API_TOKEN: Optional[str] = Field(default=None)
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")

    # Embedding Settings
    EMBEDDING_PROVIDER: str = Field(default=DEFAULT_EMBEDDING_PROVIDER)
    EMBEDDING_MODEL: str = Field(default=DEFAULT_EMBEDDING_MODEL)

    # Vector Store Settings
    VECTOR_STORE_TYPE: str = Field(default=DEFAULT_VECTOR_STORE_TYPE)
    VECTOR_STORE_DIR: str = Field(default=DEFAULT_VECTOR_STORE_DIR)

    # Ingestion Constraints
    CHUNK_SIZE: int = Field(default=DEFAULT_CHUNK_SIZE)
    CHUNK_OVERLAP: int = Field(default=DEFAULT_CHUNK_OVERLAP)
    MAX_UPLOAD_SIZE_MB: int = Field(default=DEFAULT_MAX_UPLOAD_SIZE_MB)

    @field_validator("LLM_PROVIDER")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        provider = v.lower().strip()
        if provider not in SUPPORTED_LLM_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM provider '{v}'. Choose from: {SUPPORTED_LLM_PROVIDERS}"
            )
        return provider

    @field_validator("EMBEDDING_PROVIDER")
    @classmethod
    def validate_embedding_provider(cls, v: str) -> str:
        provider = v.lower().strip()
        if provider not in SUPPORTED_EMBEDDING_PROVIDERS:
            raise ValueError(
                f"Unsupported Embedding provider '{v}'. Choose from: {SUPPORTED_EMBEDDING_PROVIDERS}"
            )
        return provider

    @field_validator("VECTOR_STORE_TYPE")
    @classmethod
    def validate_vector_store_type(cls, v: str) -> str:
        store_type = v.lower().strip()
        if store_type not in SUPPORTED_VECTOR_STORES:
            raise ValueError(
                f"Unsupported Vector Store type '{v}'. Choose from: {SUPPORTED_VECTOR_STORES}"
            )
        return store_type

    def validate_api_keys(self) -> None:
        """
        Performs logical checks on API credentials depending on config.
        """
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider.")
        if self.LLM_PROVIDER == "anthropic" and not self.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required for Anthropic provider.")
        if self.LLM_PROVIDER == "huggingface" and not self.HUGGINGFACEHUB_API_TOKEN:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN is required to run remote Hugging Face endpoints.")
        if self.EMBEDDING_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required to generate OpenAI embeddings.")


_config_cache: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """
    Returns a cached Singleton instance of the Application Config.
    """
    global _config_cache
    if _config_cache is None:
        _config_cache = AppConfig()
    return _config_cache
