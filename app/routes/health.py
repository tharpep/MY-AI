"""Health check endpoint"""

from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "personal-ai-assistant-api",
        "version": "0.1.0"
    }


@router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check including system components."""
    from ..main import gateway
    from rag.rag_setup import get_rag
    from core.config import get_config
    from app.db import get_db_connection
    
    components = {}
    overall_status = "healthy"
    
    # Check LLM Gateway
    try:
        if gateway is None:
            components["llm_gateway"] = {"status": "unhealthy", "error": "Not initialized"}
            overall_status = "unhealthy"
        else:
            providers = gateway.get_available_providers()
            components["llm_gateway"] = {
                "status": "healthy" if providers else "degraded",
                "providers": providers
            }
            if not providers:
                overall_status = "degraded"
    except Exception as e:
        components["llm_gateway"] = {"status": "unhealthy", "error": str(e)}
        overall_status = "unhealthy"
    
    # Check RAG System
    try:
        config = get_config()
        rag = get_rag()
        if config.chat_context_enabled and rag:
            stats = rag.get_stats()
            components["rag_system"] = {
                "status": "healthy",
                "documents": stats.get("document_count", 0)
            }
        elif config.chat_context_enabled:
            components["rag_system"] = {"status": "degraded", "error": "Not initialized"}
            if overall_status == "healthy":
                overall_status = "degraded"
        else:
            components["rag_system"] = {"status": "disabled"}
    except Exception as e:
        components["rag_system"] = {"status": "unhealthy", "error": str(e)}
        if overall_status == "healthy":
            overall_status = "degraded"
    
    # Check Database
    try:
        with get_db_connection() as conn:
            conn.cursor().execute("SELECT 1").fetchone()
        components["database"] = {"status": "healthy"}
    except Exception as e:
        components["database"] = {"status": "unhealthy", "error": str(e)}
        if overall_status == "healthy":
            overall_status = "degraded"
    
    return {
        "status": overall_status,
        "service": "personal-ai-assistant-api",
        "version": "0.1.0",
        "components": components
    }
