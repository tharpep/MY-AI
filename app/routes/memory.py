"""Memory Management API routes - Journal session management"""

import logging
import uuid
from typing import Optional, Any, Dict

from fastapi import APIRouter, HTTPException, Query, status

logger = logging.getLogger(__name__)

router = APIRouter()


def _generate_request_id() -> str:
    """Generate a unique request ID for tracing."""
    return f"req_{uuid.uuid4().hex[:12]}"


@router.get("/memory/sessions")
async def list_sessions(
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum sessions to return")
) -> Dict[str, Any]:
    """
    List all chat sessions.

    Returns:
        Dictionary with:
        - sessions: List of session metadata (session_id, name, message_count, last_activity, ingested_at)
        - total: Total number of sessions
        - request_id: Request ID for tracing
    """
    from core.session_store import get_session_store

    request_id = _generate_request_id()

    try:
        session_store = get_session_store()
        sessions = session_store.list_sessions(limit=limit)

        return {
            "sessions": sessions,
            "total": len(sessions),
            "request_id": request_id
        }

    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to list sessions: {str(e)}",
                    "type": "internal_error",
                    "code": "server_error"
                },
                "request_id": request_id
            }
        )


@router.get("/memory/sessions/{session_id}/messages")
async def get_session_messages(session_id: str) -> Dict[str, Any]:
    """
    Get a session with all its messages.

    Used by frontend to load a past conversation.

    Args:
        session_id: The session identifier

    Returns:
        Session metadata + messages + ingestion status
    """
    from core.session_store import get_session_store

    request_id = _generate_request_id()

    try:
        session_store = get_session_store()
        session_data = session_store.get_session_with_messages(session_id)

        if session_data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"Session not found: {session_id}",
                        "type": "not_found_error",
                        "code": "session_not_found"
                    },
                    "request_id": request_id
                }
            )

        # Get ingestion status
        from rag.rag_setup import get_rag
        rag = get_rag()
        journal = rag.journal if rag else None
        ingestion_status = journal.get_ingestion_status(session_id) if journal else {"ingested": False}

        return {
            "session_id": session_id,
            "name": session_data.get("name"),
            "created_at": session_data.get("created_at"),
            "last_activity": session_data.get("last_activity"),
            "message_count": session_data.get("message_count", 0),
            "ingestion_status": {
                "ingested": ingestion_status.get("ingested", False),
                "ingested_at": ingestion_status.get("ingested_at"),
                "has_new_messages": ingestion_status.get("has_new_messages", False),
                "chunk_count": ingestion_status.get("chunk_count", 0)
            },
            "messages": session_data.get("messages", []),
            "request_id": request_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session messages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to get session: {str(e)}",
                    "type": "internal_error",
                    "code": "server_error"
                },
                "request_id": request_id
            }
        )


@router.post("/memory/sessions/{session_id}/ingest")
async def ingest_session(session_id: str) -> Dict[str, Any]:
    """
    Manually trigger ingestion of a session into the journal RAG collection.

    Pipeline:
    1. Export session to journal_blob/
    2. Delete existing chunks from Qdrant
    3. Chunk, embed, and store in Qdrant
    4. Update ingested_at timestamp

    Args:
        session_id: The session identifier to ingest

    Returns:
        Ingestion results (chunks_created, blob_path, etc.)
    """
    from core.session_store import get_session_store

    request_id = _generate_request_id()

    try:
        # Verify session exists
        session_store = get_session_store()
        session = session_store.get_session(session_id)

        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"Session not found: {session_id}",
                        "type": "not_found_error",
                        "code": "session_not_found"
                    },
                    "request_id": request_id
                }
            )

        # Check if session has messages
        if session.get("message_count", 0) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": "Session has no messages to ingest",
                        "type": "invalid_request_error",
                        "code": "no_messages"
                    },
                    "request_id": request_id
                }
            )

        # Perform ingestion
        from rag.rag_setup import get_rag
        rag = get_rag()
        if not rag or not rag.journal:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "message": "Journal system not available. RAG may be disabled.",
                        "type": "service_unavailable",
                        "code": "journal_unavailable"
                    },
                    "request_id": request_id
                }
            )
        
        journal = rag.journal
        result = journal.ingest_session(session_id)

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "message": result["error"],
                        "type": "internal_error",
                        "code": "ingestion_failed"
                    },
                    "request_id": request_id
                }
            )

        return {
            "status": "ingested",
            "session_id": result.get("session_id"),
            "chunks_created": result.get("chunks_created", 0),
            "blob_path": result.get("blob_path"),
            "ingested_at": result.get("ingested_at"),
            "message_count": result.get("message_count", 0),
            "request_id": request_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to ingest session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to ingest session: {str(e)}",
                    "type": "internal_error",
                    "code": "server_error"
                },
                "request_id": request_id
            }
        )


@router.delete("/memory/sessions/{session_id}")
async def delete_session(session_id: str) -> Dict[str, Any]:
    """
    Delete a session and all associated data.

    Deletes from:
    - SQLite (session metadata + messages)
    - Qdrant (ingested chunks)
    - Journal blob storage (exported JSON)

    Args:
        session_id: The session identifier to delete

    Returns:
        Confirmation of deletion
    """
    request_id = _generate_request_id()

    try:
        from rag.rag_setup import get_rag
        rag = get_rag()
        if not rag or not rag.journal:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "message": "Journal system not available",
                        "type": "service_unavailable",
                        "code": "journal_unavailable"
                    },
                    "request_id": request_id
                }
            )
        
        journal = rag.journal
        success = await journal.delete_session(session_id)

        if success:
            return {
                "status": "deleted",
                "session_id": session_id,
                "request_id": request_id
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "message": "Failed to delete session",
                        "type": "internal_error",
                        "code": "delete_failed"
                    },
                    "request_id": request_id
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to delete session: {str(e)}",
                    "type": "internal_error",
                    "code": "server_error"
                },
                "request_id": request_id
            }
        )


@router.get("/memory/stats")
async def get_memory_stats() -> Dict[str, Any]:
    """
    Get statistics about journal memory.

    Returns:
        Dictionary with journal statistics including:
        - Qdrant collection stats
        - Session counts
    """
    from core.session_store import get_session_store

    request_id = _generate_request_id()

    try:
        from rag.rag_setup import get_rag
        session_store = get_session_store()
        rag = get_rag()
        
        if not rag or not rag.journal:
            # Return minimal stats if journal unavailable
            sessions = session_store.list_sessions(limit=10000)
            return {
                "journal_available": False,
                "total_sessions": len(sessions),
                "request_id": request_id
            }
        
        journal = rag.journal

        # Get Qdrant stats
        qdrant_stats = journal.get_stats()

        # Get session counts from SQLite
        all_sessions = session_store.list_sessions(limit=10000)
        sessions_needing_ingest = session_store.get_sessions_needing_ingest(limit=10000)

        return {
            "initialized": True,
            "qdrant": qdrant_stats,
            "sessions": {
                "total": len(all_sessions),
                "needing_ingest": len(sessions_needing_ingest)
            },
            "request_id": request_id
        }

    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to get stats: {str(e)}",
                    "type": "internal_error",
                    "code": "server_error"
                },
                "request_id": request_id
            }
        )


@router.get("/memory/sessions/{session_id}/status")
async def get_session_status(session_id: str) -> Dict[str, Any]:
    """
    Get the ingestion status for a specific session.

    Args:
        session_id: The session identifier

    Returns:
        Ingestion status details
    """
    from core.session_store import get_session_store

    request_id = _generate_request_id()

    try:
        session_store = get_session_store()
        session = session_store.get_session(session_id)

        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"Session not found: {session_id}",
                        "type": "not_found_error",
                        "code": "session_not_found"
                    },
                    "request_id": request_id
                }
            )

        from rag.rag_setup import get_rag
        rag = get_rag()
        journal = rag.journal if rag else None
        status_info = journal.get_ingestion_status(session_id) if journal else {"ingested": False}

        return {
            **status_info,
            "request_id": request_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to get status: {str(e)}",
                    "type": "internal_error",
                    "code": "server_error"
                },
                "request_id": request_id
            }
        )
