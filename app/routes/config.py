"""Config routes — read and runtime-patch application configuration."""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from core.config import get_config as load_config

logger = logging.getLogger(__name__)

router = APIRouter()


class ConfigUpdateRequest(BaseModel):
    # Chat context
    chat_context_enabled: Optional[bool] = None
    chat_kb_enabled: Optional[bool] = None
    chat_kb_top_k: Optional[int] = Field(None, ge=1, le=100)
    chat_kb_similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    chat_kb_use_cache: Optional[bool] = None

    # Hybrid search
    hybrid_sparse_weight: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Reranking
    rerank_enabled: Optional[bool] = None
    rerank_candidates: Optional[int] = Field(None, ge=5, le=100)

    # Query expansion
    query_expansion_enabled: Optional[bool] = None

    # KB ingestion
    kb_chunk_size: Optional[int] = Field(None, ge=100, le=5000)
    kb_chunk_overlap: Optional[int] = Field(None, ge=0, le=500)


@router.get("/config")
async def get_config() -> Dict[str, Any]:
    """Get current application configuration. API keys are masked."""
    try:
        cfg = load_config()
        return {
            "config": {
                # Gateway
                "api_gateway_url": cfg.api_gateway_url,
                "default_model": cfg.default_model,

                # Embeddings
                "embedding_model": cfg.embedding_model,
                "voyage_api_key_set": bool(cfg.voyage_api_key),

                # Hybrid search
                "hybrid_sparse_weight": cfg.hybrid_sparse_weight,

                # Reranking
                "rerank_enabled": cfg.rerank_enabled,
                "rerank_candidates": cfg.rerank_candidates,
                "rerank_model": cfg.rerank_model,

                # Query expansion
                "query_expansion_enabled": cfg.query_expansion_enabled,

                # KB ingestion
                "kb_chunk_size": cfg.kb_chunk_size,
                "kb_chunk_overlap": cfg.kb_chunk_overlap,

                # Chat context
                "chat_context_enabled": cfg.chat_context_enabled,
                "chat_kb_enabled": cfg.chat_kb_enabled,
                "chat_kb_top_k": cfg.chat_kb_top_k,
                "chat_kb_similarity_threshold": cfg.chat_kb_similarity_threshold,
                "chat_kb_use_cache": cfg.chat_kb_use_cache,

                # Logging
                "debug": cfg.debug,
            }
        }
    except Exception as e:
        logger.exception("Failed to get config")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.patch("/config")
async def update_config(request: ConfigUpdateRequest) -> Dict[str, Any]:
    """
    Update configuration values at runtime.
    Changes are NOT persisted to .env — restart reverts them.
    """
    try:
        cfg = load_config()
        update_data = request.model_dump(exclude_unset=True)

        updated = []
        for field, value in update_data.items():
            if hasattr(cfg, field):
                setattr(cfg, field, value)
                updated.append(field)

        if not updated:
            return {"updated": False, "message": "No fields to update"}

        return {
            "updated": True,
            "fields": updated,
            "message": f"Updated {len(updated)} field(s). Restart to persist via .env.",
        }
    except Exception as e:
        logger.exception("Failed to update config")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
