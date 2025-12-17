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
        default="llama3.2:1b",
        description="Default Ollama model (local)"
    )
    
    # Legacy: model_default for backward compatibility (used for Ollama)
    model_default: str = Field(
        default="llama3.2:1b",
        description="Legacy default model (use provider-specific models instead)"
    )
    
    # Previously tested Ollama models (commented for reference):
    # - llama3.2:1b (current default - lightweight, fast)
    # - qwen3:8b (larger model, better quality, requires more resources)
    # - qwen3:1.7b (medium size, balanced performance)

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
        default="./data/preindex_blob",
        description="Path to blob storage directory for uploaded files pending indexing"
    )

    # ===== RAG Configuration =====
    rag_use_persistent: bool = Field(
        default=True,
        description="Use persistent vector storage (True) or in-memory (False)"
    )
    rag_collection_name: str = Field(
        default="simrag_docs",
        description="Qdrant collection name for RAG documents"
    )
    rag_clear_on_ingest: bool = Field(
        default=True,
        description="Clear collection before ingesting new documents"
    )
    rag_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve for RAG (1-20)"
    )
    rag_similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for retrieval (0.0-1.0)"
    )
    rag_max_tokens: int = Field(
        default=200,
        ge=50,
        le=500,
        description="Maximum tokens in RAG response (50-500)"
    )
    rag_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Temperature for RAG generation (0.0-1.0)"
    )
    
    # ===== RAG Chunking Configuration =====
    rag_chunk_size: int = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Maximum characters per chunk when processing documents (100-5000)"
    )
    rag_chunk_overlap: int = Field(
        default=100,
        ge=0,
        le=500,
        description="Overlap characters between chunks (0-500)"
    )
    
    # ===== Embedding Model Configuration =====
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    
    # ===== Worker Configuration =====
    worker_job_timeout: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Maximum seconds for a worker job before timeout (30-3600)"
    )

    # ===== Chat RAG Configuration =====
    chat_rag_enabled: bool = Field(
        default=True,
        description="Enable RAG context retrieval in chat completions endpoint"
    )
    chat_rag_top_k: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Top-k documents to retrieve for chat RAG (1-100)"
    )
    chat_rag_similarity_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for chat RAG context (0.0-1.0). Lower values allow more documents through."
    )
    
    # ===== RAG Enhancement Configuration =====
    chat_rag_use_context_cache: bool = Field(
        default=True,
        description="Enable context caching to reuse RAG results from previous messages"
    )
    chat_rag_use_conversation_aware: bool = Field(
        default=True,
        description="Use conversation history to build better RAG queries"
    )
    chat_rag_min_results_threshold: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Minimum number of RAG results before trying fallback strategies (0-10)"
    )

    # ===== Document Source Configuration =====
    rag_use_documents_folder: bool = Field(
        default=False,
        description="Use data/documents folder instead of data/corpus for RAG ingestion (True = documents, False = corpus)"
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
        elif self.provider_name == "ollama":
            return self.model_ollama
        else:
            # Fallback to legacy model_default
            return self.model_default
    
    def get_model_for_provider(self, provider: str) -> str:
        """Get default model for a specific provider"""
        if provider == "anthropic":
            return self.model_anthropic
        elif provider == "purdue":
            return self.model_purdue
        elif provider == "ollama":
            return self.model_ollama
        else:
            return self.model_default

    @property
    def use_ollama(self) -> bool:
        """Check if using Ollama provider"""
        return self.provider_type == "local" and self.provider_name == "ollama"

    def _get_model_suffix(self) -> str:
        """Extract model size suffix from model name for directory naming"""
        model_lower = self.model_default.lower()
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
    def rag_documents_folder(self) -> str:
        """Get the documents folder path based on config"""
        if self.rag_use_documents_folder:
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


# Backward compatibility functions
def get_rag_config() -> AppConfig:
    """Get RAG configuration (backward compatibility - returns full config)"""
    return get_config()


def get_tuning_config() -> AppConfig:
    """Get tuning configuration (backward compatibility - returns full config)"""
    return get_config()

