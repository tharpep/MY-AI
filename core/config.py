"""
Primary Application Configuration
Unified configuration using Pydantic Settings for type safety and .env support
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices
from typing import Literal, Optional
from functools import lru_cache


class AppConfig(BaseSettings):
    """Primary application configuration - controls model and gateway settings"""

    # ===== Provider Configuration =====
    provider_type: Literal["local", "purdue", "external"] = Field(
        default="local",
        description="Type of AI provider: local (Ollama), purdue (Purdue GenAI), or external (OpenAI/Anthropic)"
    )
    provider_name: str = Field(
        default="ollama",
        description="Specific provider name: ollama, purdue, openai, anthropic, etc."
    )
    provider_fallback: Optional[str] = Field(
        default=None,
        description="Optional fallback provider if primary fails"
    )

    # ===== Model Configuration =====
    # Provider-specific default models
    model_anthropic: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Default Anthropic/Claude model (latest Haiku 4.5)"
    )
    model_purdue: str = Field(
        default="llama3.1:latest",
        description="Default Purdue GenAI Studio model"
    )
    model_ollama: str = Field(
        default="llama3.1:8b",
        description="Default Ollama model (local)"
    )

    # ===== Ollama Configuration =====
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL"
    )
    ollama_timeout: float = Field(
        default=5.0,
        description="Ollama connection timeout in seconds"
    )

    # ===== Purdue API Configuration =====
    purdue_api_key: Optional[str] = Field(
        default=None,
        description="Purdue GenAI Studio API key (from .env)"
    )

    # ===== External API Configuration (for future use) =====
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key (from .env)"
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL"
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key (from .env as CLAUDE or ANTHROPIC_API_KEY)",
        validation_alias=AliasChoices("CLAUDE", "ANTHROPIC_API_KEY")
    )

    # ===== Qdrant Configuration =====
    qdrant_host: str = Field(
        default="localhost",
        description="Qdrant server host"
    )
    qdrant_port: int = Field(
        default=6333,
        ge=1,
        le=65535,
        description="Qdrant server port"
    )

    # ===== Redis Configuration =====
    redis_host: str = Field(
        default="localhost",
        description="Redis server host"
    )
    redis_port: int = Field(
        default=6379,
        ge=1,
        le=65535,
        description="Redis server port"
    )

    # ===== Blob Storage Configuration =====
    blob_storage_path: str = Field(
        default="./data/library_blob",
        description="Path to blob storage directory for Library documents"
    )
    # ===== Vector Storage Configuration (shared by Library + Journal) =====
    storage_use_persistent: bool = Field(
        default=True,
        description="Use persistent Qdrant server (True) or in-memory (False)"
    )
    
    # ===== Library Configuration =====
    library_collection_name: str = Field(
        default="library_docs",
        description="Qdrant collection name for Library documents"
    )
    library_clear_on_ingest: bool = Field(
        default=True,
        description="Clear collection before ingesting new documents"
    )
    library_chunk_size: int = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Maximum characters per chunk when processing documents (100-5000)"
    )
    library_chunk_overlap: int = Field(
        default=100,
        ge=0,
        le=500,
        description="Overlap characters between chunks (0-500)"
    )
    
    # ===== Journal Configuration =====
    journal_collection_name: str = Field(
        default="journal_entries",
        description="Qdrant collection name for Journal (chat history)"
    )
    journal_blob_storage_path: str = Field(
        default="./data/journal_blob",
        description="Path to blob storage directory for exported Journal sessions"
    )
    journal_title_max_length: int = Field(
        default=25,
        ge=10,
        le=100,
        description="Maximum length for auto-generated session titles (10-100)"
    )
    journal_chunk_size: int = Field(
        default=1500,
        ge=100,
        le=5000,
        description="Maximum characters per chunk when ingesting journal sessions (100-5000)"
    )
    journal_chunk_overlap: int = Field(
        default=150,
        ge=0,
        le=500,
        description="Overlap characters between journal chunks (0-500)"
    )
    
    # ===== Embedding Model Configuration =====
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="Sentence transformer model for embeddings (used by both Library and Journal)"
    )
    
    # ===== Hybrid Search Configuration =====
    hybrid_sparse_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for sparse vectors in hybrid search (0=dense only, 1=sparse only)"
    )
    
    # ===== Mnemosyne: Hardware Selection =====
    hardware_mode: Literal["gpu", "cpu", "auto"] = Field(
        default="auto",
        description="Hardware mode: 'gpu' for GPU-preferred models, 'cpu' for lightweight models, 'auto' for detection"
    )
    
    # ===== Worker Configuration =====
    worker_job_timeout: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Maximum seconds for a worker job before timeout (30-3600)"
    )

    # ===== Chat Context Configuration =====
    # Master switch for all context injection
    chat_context_enabled: bool = Field(
        default=True,
        description="Master switch: Enable context injection (Library + Journal) in chat"
    )
    
    # Library (document knowledge) settings
    chat_library_enabled: bool = Field(
        default=True,
        description="Enable Library (document RAG) context retrieval in chat"
    )
    chat_library_top_k: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Top-k documents to retrieve from Library (1-100)"
    )
    chat_library_similarity_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for Library context (0.0-1.0)"
    )
    chat_library_use_cache: bool = Field(
        default=True,
        description="Enable caching to reuse Library results from similar queries"
    )
    
    # Journal (chat history) settings
    chat_journal_enabled: bool = Field(
        default=False,
        description="Enable Journal (chat history) context retrieval in chat"
    )
    chat_journal_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Top-k entries to retrieve from Journal (1-50)"
    )

    # ===== Document Source Configuration =====
    library_use_documents_folder: bool = Field(
        default=False,
        description="Use data/documents folder instead of data/corpus for Library ingestion (True = documents, False = corpus)"
    )

    # ===== Logging/Output Configuration =====
    log_output: bool = Field(
        default=True,
        description="Enable verbose logging/output for RAG retrieval, model usage, and system operations (True = verbose, False = clean)"
    )

    # ===== Tuning Configuration =====
    tuning_device: str = Field(
        default="auto",
        description="Device for tuning: auto, cpu, cuda, mps"
    )
    tuning_max_length: int = Field(
        default=512,
        ge=256,
        le=1024,
        description="Maximum sequence length for tuning (256-1024)"
    )
    tuning_num_epochs: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of training epochs (1-10)"
    )
    tuning_batch_size: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Batch size for training (1-16)"
    )
    tuning_learning_rate: float = Field(
        default=5e-5,
        description="Learning rate for training"
    )
    tuning_version: str = Field(
        default="v1.0",
        description="Model version (e.g., v1.0, v1.1, v2.0)"
    )
    tuning_create_version_dir: bool = Field(
        default=True,
        description="Create versioned subdirectories for tuned models"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # ===== Computed Properties =====
    @property
    def model_name(self) -> str:
        """Get model name based on current provider (returns provider-specific default)"""
        # Return provider-specific model based on provider_name
        if self.provider_name == "anthropic":
            return self.model_anthropic
        elif self.provider_name == "purdue":
            return self.model_purdue
        else:
            # Default to Ollama model
            return self.model_ollama
    
    def get_model_for_provider(self, provider: str) -> str:
        """Get default model for a specific provider"""
        if provider == "anthropic":
            return self.model_anthropic
        elif provider == "purdue":
            return self.model_purdue
        else:
            return self.model_ollama

    @property
    def use_ollama(self) -> bool:
        """Check if using Ollama provider"""
        return self.provider_type == "local" and self.provider_name == "ollama"

    def _get_model_suffix(self) -> str:
        """Extract model size suffix from model name for directory naming"""
        model_lower = self.model_ollama.lower()
        if "1b" in model_lower:
            return "1b"
        elif "8b" in model_lower:
            return "8b"
        elif "1.7b" in model_lower:
            return "1.7b"
        return "default"

    @property
    def output_dir(self) -> str:
        """Get output directory for tuned models"""
        model_suffix = self._get_model_suffix()
        base_dir = f"./tuned_models/{model_suffix}"
        if self.tuning_create_version_dir:
            return f"{base_dir}/{self.tuning_version}"
        return base_dir

    @property
    def model_registry_path(self) -> str:
        """Get path to model registry"""
        model_suffix = self._get_model_suffix()
        return f"./tuned_models/{model_suffix}/model_registry.json"

    @property
    def library_documents_folder(self) -> str:
        """Get the documents folder path based on config"""
        if self.library_use_documents_folder:
            return "./data/documents"
        else:
            return "./data/corpus"


# Global config instance (singleton pattern)
_config: Optional[AppConfig] = None


@lru_cache()
def get_config() -> AppConfig:
    """
    Get application configuration (singleton, cached)
    
    Returns:
        AppConfig: Application configuration instance
    """
    global _config
    if _config is None:
        _config = AppConfig()
    return _config




