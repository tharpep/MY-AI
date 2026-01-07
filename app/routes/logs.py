"""
Logs API routes - Endpoint for retrieving request logs

Provides:
- GET /v1/logs - Get recent request logs
"""

import uuid
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status

from ..db import get_recent_requests, get_request_by_id

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/logs")
async def get_logs(
    limit: int = Query(100, ge=1, le=1000, description="Number of logs to return"),
    offset: int = Query(0, ge=0, description="Number of logs to skip")
) -> Dict[str, Any]:
    """
    Get recent request logs.
    
    Returns a list of requests logged by the system, ordered by timestamp descending.
    Includes details like endpoint, status code, response time, and token usage.
    """
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    
    try:
        logs = get_recent_requests(limit=limit, offset=offset)
        
        return {
            "object": "list",
            "data": logs,
            "meta": {
                "limit": limit,
                "offset": offset,
                "count": len(logs)
            },
            "request_id": request_id
        }
        
    except Exception as e:
        logger.error(f"[Logs] Failed to get logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to get logs: {str(e)}",
                    "type": "internal_error",
                    "code": "server_error"
                },
                "request_id": request_id
            }
        )

@router.get("/logs/{log_id}")
async def get_log_detail(log_id: str) -> Dict[str, Any]:
    """
    Get details for a specific log entry by request_id.
    """
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    
    try:
        log = get_request_by_id(log_id)
        
        if not log:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"Log entry not found: {log_id}",
                        "type": "not_found",
                        "code": "log_not_found"
                    },
                    "request_id": request_id
                }
            )
            
        return {
            "object": "log_entry",
            "data": log,
            "request_id": request_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Logs] Failed to get log detail: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to get log detail: {str(e)}",
                    "type": "internal_error",
                    "code": "server_error"
                },
                "request_id": request_id
            }
        )
