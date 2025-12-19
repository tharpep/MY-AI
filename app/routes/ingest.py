"""Document Ingestion API routes - Ingest documents into RAG system"""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, UploadFile, status

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ingest")
async def ingest_documents(folder_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Ingest documents into the RAG system.
    
    Args:
        folder_path: Path to folder containing documents to ingest. 
                     If not provided, uses config.library_documents_folder (corpus or documents based on library_use_documents_folder)
        
    Returns:
        Dictionary with ingestion results including:
        - success: Whether ingestion was successful
        - processed: Number of files processed
        - failed: Number of files that failed
        - files: List of file processing details
        - errors: List of any errors encountered
        - request_id: Request ID for tracing
    """
    from rag.rag_setup import ContextEngine
    from rag.document_ingester import DocumentIngester
    from core.config import get_config
    
    # Generate request ID for tracing
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    
    try:
        # Use config folder if not provided
        config = get_config()
        if folder_path is None:
            folder_path = config.library_documents_folder
        
        # Validate folder path
        folder = Path(folder_path)
        if not folder.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"Folder not found: {folder_path}",
                        "type": "not_found_error",
                        "code": "folder_not_found"
                    },
                    "request_id": request_id
                }
            )
        
        if not folder.is_dir():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": f"Path is not a directory: {folder_path}",
                        "type": "invalid_request_error",
                        "code": "not_a_directory"
                    },
                    "request_id": request_id
                }
            )
        
        # Initialize RAG system
        rag = ContextEngine()
        
        # Initialize ingester
        ingester = DocumentIngester(rag)
        
        # Ingest documents
        result = ingester.ingest_folder(folder_path)
        
        return {
            "success": result["success"],
            "processed": result["processed"],
            "failed": result["failed"],
            "files": result["files"],
            "errors": result.get("errors", []),
            "request_id": request_id
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (already properly formatted)
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Document ingestion failed: {str(e)}",
                    "type": "internal_error",
                    "code": "server_error"
                },
                "request_id": request_id
            }
        )


@router.post("/ingest/upload")
async def upload_document(file: UploadFile) -> Dict[str, Any]:
    """
    Upload a document for async processing into RAG system.
    
    File is saved to blob storage and queued for background processing.
    
    Args:
        file: The uploaded file (TXT, MD, PDF, or DOCX)
        
    Returns:
        Dictionary with:
        - job_id: Job identifier for tracking processing status
        - blob_id: Blob identifier for the stored file
        - status: Current job status (queued)
        - request_id: Request ID for tracing
    """
    from core.file_storage import get_blob_storage
    from core.queue import get_redis_queue
    from rag.document_parser import DocumentParser
    
    # Generate request ID for tracing
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    logger.info(f"[Ingest] Upload request received: {file.filename} (request_id={request_id})")
    
    try:
        # Validate file type
        filename = file.filename or "unknown"
        parser = DocumentParser()
        if not parser.supports(Path(filename)):
            logger.warning(f"[Ingest] Unsupported file type: {filename}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": f"Unsupported file type: {filename}. Supported: .txt, .md, .pdf, .docx",
                        "type": "invalid_request_error",
                        "code": "unsupported_file_type"
                    },
                    "request_id": request_id
                }
            )
        
        # Read file content
        content = await file.read()
        logger.info(f"[Ingest] File read: {filename} ({len(content)} bytes)")
        
        # Save to blob storage
        storage = get_blob_storage()
        blob_id = storage.save(content, filename)
        logger.info(f"[Ingest] Saved to blob storage: {blob_id}")
        
        # Enqueue for processing
        queue = await get_redis_queue()
        job_id = await queue.enqueue('process_document', blob_id)
        logger.info(f"[Ingest] Job enqueued: job_id={job_id}, blob_id={blob_id}")
        
        return {
            "job_id": job_id,
            "blob_id": blob_id,
            "filename": filename,
            "status": "queued",
            "request_id": request_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Check if it's a Redis/queue connection issue
        error_str = str(e).lower()
        if any(term in error_str for term in ["redis", "connection", "refused", "timeout"]):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "message": "Queue service unavailable. Redis may not be running.",
                        "type": "service_unavailable",
                        "code": "queue_unavailable"
                    },
                    "request_id": request_id
                }
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Upload failed: {str(e)}",
                    "type": "internal_error",
                    "code": "server_error"
                },
                "request_id": request_id
            }
        )



@router.post("/ingest/file/{filename}")
async def ingest_manual_file(filename: str) -> Dict[str, Any]:
    """
    Manually trigger ingestion for a file.
    
    Can process:
    1. A raw file manually placed in the library_blob folder (will be imported/adopted)
    2. An existing blob_id (re-ingest)
    3. An existing file by original filename (re-ingest)
    
    Args:
        filename: Name of file or blob_id to ingest
        
    Returns:
        Job queue status
    """
    from core.file_storage import get_blob_storage
    from core.queue import get_redis_queue
    
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    logger.info(f"[Ingest] Manual ingest request: {filename} ({request_id})")
    
    try:
        storage = get_blob_storage()
        blob_id = None
        
        # Strategy 1: Check if it's a valid blob_id directly
        if storage.get_info(filename):
            blob_id = filename
            logger.info(f"[Ingest] Found existing blob_id: {blob_id}")
            
        # Strategy 2: Check if it's a raw file in the directory (adopt it)
        elif (storage.storage_path / filename).exists() and (storage.storage_path / filename).is_file():
            raw_path = storage.storage_path / filename
            logger.info(f"[Ingest] Found raw file, adopting: {filename}")
            
            # Read and save as proper blob
            content = raw_path.read_bytes()
            blob_id = storage.save(content, filename)
            
            # Delete original raw file to prevent duplicates/confusion
            try:
                raw_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete adopted file {filename}: {e}")
                
        # Strategy 3: Search manifests for original_filename
        else:
            blobs = storage.list()
            # Find matches, sort by recent first
            matches = [b for b in blobs if b.original_filename == filename]
            if matches:
                # Use most recent
                matches.sort(key=lambda x: x.created_at, reverse=True)
                blob_id = matches[0].blob_id
                logger.info(f"[Ingest] Found blob by filename: {filename} -> {blob_id}")
        
        # Check if we found anything
        if not blob_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"File not found: {filename}. Not in storage or index.",
                        "type": "not_found_error",
                        "code": "file_not_found"
                    },
                    "request_id": request_id
                }
            )
            
        # Enqueue for processing
        queue = await get_redis_queue()
        job_id = await queue.enqueue('process_document', blob_id)
        logger.info(f"[Ingest] Job enqueued: job_id={job_id}, blob_id={blob_id}")
        
        return {
            "job_id": job_id,
            "blob_id": blob_id,
            "filename": filename,
            "status": "queued",
            "request_id": request_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Check for Redis connection issues (common in dev)
        if "redis" in str(e).lower() or "connection" in str(e).lower():
             raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "message": "Queue service unavailable. Redis may not be running.",
                        "type": "service_unavailable",
                        "code": "queue_unavailable"
                    },
                    "request_id": request_id
                }
            )
            
        logger.error(f"[Ingest] Manual ingest failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Ingestion failed: {str(e)}",
                    "type": "internal_error",
                    "code": "server_error"
                },
                "request_id": request_id
            }
        )


@router.get("/ingest/jobs/{job_id}")
async def get_job_status(job_id: str) -> Dict[str, Any]:
    """
    Get the status of an ingestion job.
    
    Args:
        job_id: The job identifier returned from upload
        
    Returns:
        Dictionary with job status details
    """
    from core.queue import get_redis_queue
    
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    
    try:
        queue = await get_redis_queue()
        job_status = await queue.get_job_status(job_id)
        
        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"Job not found: {job_id}",
                        "type": "not_found_error",
                        "code": "job_not_found"
                    },
                    "request_id": request_id
                }
            )
        
        return {
            "job_id": job_status.job_id,
            "status": job_status.status,
            "created_at": job_status.created_at,
            "completed_at": job_status.completed_at,
            "error": job_status.error,
            "request_id": request_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to get job status: {str(e)}",
                    "type": "internal_error",
                    "code": "server_error"
                },
                "request_id": request_id
            }
        )


@router.get("/ingest/blobs")
async def list_blobs() -> Dict[str, Any]:
    """
    List all staged blobs in storage.
    
    Returns:
        Dictionary with list of blob info
    """
    from core.file_storage import get_blob_storage
    
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    
    try:
        storage = get_blob_storage()
        blobs = storage.list()
        
        return {
            "blobs": [
                {
                    "blob_id": b.blob_id,
                    "original_filename": b.original_filename,
                    "file_extension": b.file_extension,
                    "size_bytes": b.size_bytes,
                    "created_at": b.created_at
                }
                for b in blobs
            ],
            "count": len(blobs),
            "request_id": request_id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to list blobs: {str(e)}",
                    "type": "internal_error",
                    "code": "server_error"
                },
                "request_id": request_id
            }
        )


@router.delete("/ingest/blobs/{blob_id}")
async def delete_blob(blob_id: str) -> Dict[str, Any]:
    """
    Delete a blob from storage.
    
    Args:
        blob_id: The blob identifier
        
    Returns:
        Dictionary confirming deletion
    """
    from core.file_storage import get_blob_storage
    
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    
    try:
        storage = get_blob_storage()
        deleted = storage.delete(blob_id)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": {
                        "message": f"Blob not found: {blob_id}",
                        "type": "not_found_error",
                        "code": "blob_not_found"
                    },
                    "request_id": request_id
                }
            )
        
        return {
            "deleted": True,
            "blob_id": blob_id,
            "request_id": request_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to delete blob: {str(e)}",
                    "type": "internal_error",
                    "code": "server_error"
                },
                "request_id": request_id
            }
        )


@router.get("/ingest/indexed")
async def get_indexed_stats() -> Dict[str, Any]:
    """
    Get statistics about indexed documents in Qdrant.
    
    Returns:
        Dictionary with collection info and document counts
    """
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    
    try:
        from rag.rag_setup import get_rag
        rag = get_rag()
        stats = rag.get_stats()
        
        return {
            "collection": stats.get("collection_name", "unknown"),
            "total_documents": stats.get("document_count", 0),
            "vector_dimension": stats.get("vector_size", 0),
            "storage_type": "server" if rag.vector_store.use_persistent else "in-memory",
            "request_id": request_id
        }
        
    except Exception as e:
        logger.error(f"[Ingest] Failed to get indexed stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to get indexed stats: {str(e)}",
                    "type": "internal_error",
                    "code": "server_error"
                },
                "request_id": request_id
            }
        )


@router.delete("/ingest/indexed")
async def clear_all_indexed(
    confirm: bool = Query(default=False, description="Confirm deletion of ALL indexed documents")
) -> Dict[str, Any]:
    """
    Clear all indexed documents from Qdrant.
    
    Args:
        confirm: Must be set to true to execute deletion
    
    Returns:
        Dictionary with clear result
    """
    from fastapi import Query
    from rag.rag_setup import get_rag
    
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "message": "Deletion requires confirmation. Set ?confirm=true",
                    "type": "invalid_request_error",
                    "code": "missing_confirmation"
                },
                "request_id": request_id
            }
        )
    
    try:
        rag = get_rag()
        result = rag.clear_collection()
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "message": result["error"],
                        "type": "internal_error",
                        "code": "server_error"
                    },
                    "request_id": request_id
                }
            )
        
        return {
            "cleared": True,
            "message": result.get("message", "Collection cleared"),
            "request_id": request_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Ingest] Failed to clear indexed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to clear indexed: {str(e)}",
                    "type": "internal_error",
                    "code": "server_error"
                },
                "request_id": request_id
            }
        )


@router.get("/ingest/indexed/files")
async def list_indexed_files() -> Dict[str, Any]:
    """
    List all indexed files with their chunk counts.
    
    Returns:
        Dictionary with list of indexed files
    """
    from rag.rag_setup import get_rag
    
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    
    try:
        rag = get_rag()
        result = rag.get_indexed_files()
        
        return {
            "files": result.get("files", []),
            "total_files": result.get("total_files", 0),
            "request_id": request_id
        }
        
    except Exception as e:
        logger.error(f"[Ingest] Failed to list indexed files: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to list indexed files: {str(e)}",
                    "type": "internal_error",
                    "code": "server_error"
                },
                "request_id": request_id
            }
        )


@router.delete("/ingest/indexed/{blob_id}")
async def delete_indexed_file(blob_id: str) -> Dict[str, Any]:
    """
    Delete all indexed chunks for a specific file (blob_id).
    
    Args:
        blob_id: The blob identifier to delete
        
    Returns:
        Dictionary with deletion result
    """
    from rag.rag_setup import get_rag
    
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    
    try:
        rag = get_rag()
        result = rag.delete_by_blob_id(blob_id)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": {
                        "message": result["error"],
                        "type": "internal_error",
                        "code": "server_error"
                    },
                    "request_id": request_id
                }
            )
        
        return {
            "deleted": True,
            "blob_id": blob_id,
            "message": result.get("message", "Chunks deleted"),
            "request_id": request_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Ingest] Failed to delete indexed file {blob_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": f"Failed to delete indexed file: {str(e)}",
                    "type": "internal_error",
                    "code": "server_error"
                },
                "request_id": request_id
            }
        )
