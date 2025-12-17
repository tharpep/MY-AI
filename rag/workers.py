"""
RAG Document Processing Workers
Async workers for document ingestion via arq
"""

import logging
from arq.connections import RedisSettings

logger = logging.getLogger(__name__)


async def process_document(ctx: dict, blob_id: str) -> dict:
    """
    Process a document from blob storage into Qdrant.
    
    This is the main worker function called by arq.
    
    Args:
        ctx: arq context (contains redis connection)
        blob_id: The blob to process
        
    Returns:
        dict with processing results
    """
    from core.file_storage import get_blob_storage
    from rag.document_parser import get_document_parser
    from rag.document_ingester import DocumentIngester
    from rag.rag_setup import BasicRAG
    
    logger.info(f"Processing document: {blob_id}")
    
    # Get blob path
    storage = get_blob_storage()
    file_path = storage.get(blob_id)
    
    if file_path is None:
        raise ValueError(f"Blob not found: {blob_id}")
    
    # Parse document
    parser = get_document_parser()
    parsed = parser.parse(file_path)
    
    if parsed is None:
        raise ValueError(f"Failed to parse document: {file_path}")
    
    # Initialize RAG and ingest
    rag = BasicRAG()
    ingester = DocumentIngester(rag)
    
    # Chunk and ingest the parsed text
    chunks = ingester._chunk_text(parsed.text)
    count = rag.add_documents(chunks)
    
    logger.info(f"Indexed {count} chunks from {blob_id}")
    
    return {
        "blob_id": blob_id,
        "chunks_indexed": count,
        "file_type": parsed.file_type,
        "original_filename": parsed.original_filename
    }


# arq worker settings - used when running: arq rag.workers.WorkerSettings
class WorkerSettings:
    """arq worker configuration"""
    functions = [process_document]
    redis_settings = RedisSettings(host='localhost', port=6379)
    max_jobs = 10
    job_timeout = 300  # 5 minutes per job
