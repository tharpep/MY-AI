"""
Config API routes - Endpoints for reading and updating application configuration

Provides:
- GET /v1/config - Get current configuration values
- PATCH /v1/config - Update configuration values
"""

import uuid
import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


# ===== Request/Response Models =====

class ConfigUpdateRequest(BaseModel):
    """Request body for updating configuration values"""
    
    # Provider settings
    provider_name: Optional[str] = Field(None, description="AI provider: ollama, purdue, anthropic")
    
    # Model settings
    model_ollama: Optional[str] = Field(None, description="Default Ollama model")
    model_purdue: Optional[str] = Field(None, description="Default Purdue model")
    model_anthropic: Optional[str] = Field(None, description="Default Anthropic model")
    # Context settings (Library + Journal)
    chat_context_enabled: Optional[bool] = Field(None, description="Master switch for context injection")
    chat_library_enabled: Optional[bool] = Field(None, description="Enable Library (document) context")
    chat_library_top_k: Optional[int] = Field(None, ge=1, le=100, description="Top-k docs from Library")
    chat_library_similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similarity threshold")
    chat_journal_enabled: Optional[bool] = Field(None, description="Enable Journal (chat history) context")
    chat_journal_top_k: Optional[int] = Field(None, ge=1, le=50, description="Top-k entries from Journal")
    library_chunk_size: Optional[int] = Field(None, ge=100, le=5000, description="Chunk size for indexing")
    library_chunk_overlap: Optional[int] = Field(None, ge=0, le=500, description="Chunk overlap")
    
    # Embedding
    embedding_model: Optional[str] = Field(None, description="Embedding model name")
    
    # Infrastructure (read-only in most cases, but includable)
    qdrant_host: Optional[str] = Field(None, description="Qdrant host")
    qdrant_port: Optional[int] = Field(None, ge=1, le=65535, description="Qdrant port")
    redis_host: Optional[str] = Field(None, description="Redis host")
    redis_port: Optional[int] = Field(None, ge=1, le=65535, description="Redis port")
    
    # Logging
    log_output: Optional[bool] = Field(None, description="Enable verbose logging")


# ===== Config Endpoints =====

@router.get("/config")
async def get_config() -> Dict[str, Any]:
    """
    Get current application configuration.
    
    Returns all configurable settings that can be modified by the frontend.
    Sensitive values (API keys) are masked.
    """
    from core.config import get_config as load_config
    
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    
    try:
        config = load_config()
        
        # Return config values (mask sensitive data)
        return {
            "config": {
                # Provider
                "provider_type": config.provider_type,
                "provider_name": config.provider_name,
                "provider_fallback": config.provider_fallback,
                
                # Models
                "model_ollama": config.model_ollama,
                "model_purdue": config.model_purdue,
                "model_anthropic": config.model_anthropic,
                
                # Context - Library + Journal
                "chat_context_enabled": config.chat_context_enabled,
                "chat_library_enabled": config.chat_library_enabled,
                "chat_library_top_k": config.chat_library_top_k,
                "chat_library_similarity_threshold": config.chat_library_similarity_threshold,
                "chat_library_use_cache": config.chat_library_use_cache,
                "chat_journal_enabled": config.chat_journal_enabled,
                "chat_journal_top_k": config.chat_journal_top_k,
                
                # Library - Indexing
                "library_collection_name": config.library_collection_name,
                "library_chunk_size": config.library_chunk_size,
                "library_chunk_overlap": config.library_chunk_overlap,
                "storage_use_persistent": config.storage_use_persistent,
                
                # Embedding
                "embedding_model": config.embedding_model,
                
                # Hardware
                "hardware_mode": config.hardware_mode,
                
                # Infrastructure
                "qdrant_host": config.qdrant_host,
                "qdrant_port": config.qdrant_port,
                "redis_host": config.redis_host,
                "redis_port": config.redis_port,
                "blob_storage_path": config.blob_storage_path,
                
                # Worker
                "worker_job_timeout": config.worker_job_timeout,
                
                # Logging
                "log_output": config.log_output,
                
                # API Keys (masked)
                "purdue_api_key_set": bool(config.purdue_api_key),
                "anthropic_api_key_set": bool(config.anthropic_api_key),
                "openai_api_key_set": bool(config.openai_api_key),
            },
            "request_id": request_id
        }
        
    except Exception as e:
        logger.error(f"[Config] Failed to get config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to get config: {str(e)}",
                    "type": "internal_error",
                    "code": "server_error"
                },
                "request_id": request_id
            }
        )


@router.patch("/config")
async def update_config(request: ConfigUpdateRequest) -> Dict[str, Any]:
    """
    Update application configuration values.
    
    Note: Changes are applied at runtime but NOT persisted to .env file.
    To persist changes, the frontend should display current values and
    instruct users to update their .env file.
    
    Returns the updated configuration values.
    """
    from core.config import get_config as load_config
    
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    
    try:
        config = load_config()
        updated_fields = []
        
        # Update only provided fields
        update_data = request.model_dump(exclude_unset=True)
        
        for field, value in update_data.items():
            if hasattr(config, field):
                setattr(config, field, value)
                updated_fields.append(field)
                logger.info(f"[Config] Updated {field} = {value}")
        
        if not updated_fields:
            return {
                "updated": False,
                "message": "No fields to update",
                "request_id": request_id
            }
        
        return {
            "updated": True,
            "fields": updated_fields,
            "message": f"Updated {len(updated_fields)} field(s). Note: Changes are runtime only - update .env to persist.",
            "request_id": request_id
        }
        
    except Exception as e:
        logger.error(f"[Config] Failed to update config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to update config: {str(e)}",
                    "type": "internal_error",
                    "code": "server_error"
                },
                "request_id": request_id
            }
        )


@router.get("/config/schema")
async def get_config_schema() -> Dict[str, Any]:
    """
    Get configuration schema with field descriptions and constraints.
    
    Useful for building dynamic settings UIs.
    """
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    
    # Return schema for frontend to build settings UI
    return {
        "schema": {
            "provider_name": {
                "type": "string",
                "options": ["ollama", "purdue", "anthropic"],
                "description": "AI provider to use"
            },
            "chat_context_enabled": {
                "type": "boolean",
                "description": "Master switch: Enable context injection"
            },
            "chat_library_enabled": {
                "type": "boolean",
                "description": "Enable Library (document) context"
            },
            "chat_library_top_k": {
                "type": "integer",
                "min": 1,
                "max": 100,
                "description": "Number of documents to retrieve from Library"
            },
            "chat_library_similarity_threshold": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Minimum similarity score for Library"
            },
            "chat_journal_enabled": {
                "type": "boolean",
                "description": "Enable Journal (chat history) context"
            },
            "chat_journal_top_k": {
                "type": "integer",
                "min": 1,
                "max": 50,
                "description": "Number of entries to retrieve from Journal"
            },
            "library_chunk_size": {
                "type": "integer",
                "min": 100,
                "max": 5000,
                "description": "Max characters per chunk"
            },
            "library_chunk_overlap": {
                "type": "integer",
                "min": 0,
                "max": 500,
                "description": "Overlap between chunks"
            },
            "embedding_model": {
                "type": "string",
                "description": "Sentence transformer model for embeddings"
            },
            "log_output": {
                "type": "boolean",
                "description": "Enable verbose logging"
            }
        },
        "request_id": request_id
    }
