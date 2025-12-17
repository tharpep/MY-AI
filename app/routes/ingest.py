"""Document Ingestion API routes - Ingest documents into RAG system"""

import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, UploadFile, status

router = APIRouter()


@router.post("/ingest")
async def ingest_documents(folder_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Ingest documents into the RAG system.
    
    Args:
        folder_path: Path to folder containing documents to ingest. 
                     If not provided, uses config.rag_documents_folder (corpus or documents based on rag_use_documents_folder)
        
    Returns:
        Dictionary with ingestion results including:
        - success: Whether ingestion was successful
        - processed: Number of files processed
        - failed: Number of files that failed
        - files: List of file processing details
        - errors: List of any errors encountered
        - request_id: Request ID for tracing
    """
    from rag.rag_setup import BasicRAG
    from rag.document_ingester import DocumentIngester
    from core.config import get_config
    
    # Generate request ID for tracing
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    
    try:
        # Use config folder if not provided
        config = get_config()
        if folder_path is None:
            folder_path = config.rag_documents_folder
        
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
        rag = BasicRAG()
        
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
    
    try:
        # Validate file type
        filename = file.filename or "unknown"
        parser = DocumentParser()
        if not parser.supports(Path(filename)):
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
        
        # Save to blob storage
        storage = get_blob_storage()
        blob_id = storage.save(content, filename)
        
        # Enqueue for processing
        queue = await get_redis_queue()
        job_id = await queue.enqueue('process_document', blob_id)
        
        return {
            "job_id": job_id,
            "blob_id": blob_id,
            "filename": filename,
            "status": "queued",
            "request_id": request_id
        }
        
    except HTTPException:
        raise
    except ConnectionError as e:
        # Redis not available - graceful error
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
    except Exception as e:
        # Check if it's a Redis connection issue
        error_str = str(e).lower()
        if "redis" in error_str or "connection" in error_str or "refused" in error_str:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "message": "Queue service unavailable. Please try again later.",
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
