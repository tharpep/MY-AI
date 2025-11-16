"""
Primary Application Configuration
Unified configuration using Pydantic Settings for type safety and .env support
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
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
    model_default: str = Field(
        default="llama3.2:1b",
        description="Default model to use"
    )
    
    # Previously tested models (commented for reference):
    # - llama3.2:1b (current default - lightweight, fast)
    # - qwen3:8b (larger model, better quality, requires more resources)
    # - qwen3:1.7b (medium size, balanced performance)
    # 
    # To use a different model, set MODEL_DEFAULT in .env or override model_default
    # Example: MODEL_DEFAULT=qwen3:8b

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
        description="Anthropic API key (from .env)"
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
        """Get model name (returns model_default)"""
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

