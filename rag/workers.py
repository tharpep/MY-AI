"""RAG Document Processing Workers"""

import logging
from arq.connections import RedisSettings

logger = logging.getLogger(__name__)


async def process_document(ctx: dict, blob_id: str) -> dict:
    """Process a document from blob storage into Qdrant."""
    import traceback
    
    try:
        from core.file_storage import get_blob_storage
        from rag.document_parser import get_document_parser
        from rag.document_ingester import DocumentIngester
        from rag.rag_setup import get_rag
        
        logger.info(f"[Worker] Starting processing for blob: {blob_id}")
        
        storage = get_blob_storage()
        file_path = storage.get(blob_id)
        blob_info = storage.get_info(blob_id)
        
        if file_path is None:
            logger.error(f"[Worker] Blob not found: {blob_id}")
            raise ValueError(f"Blob not found: {blob_id}")
        
        original_filename = blob_info.original_filename if blob_info else file_path.name
        logger.info(f"[Worker] Found blob at: {file_path} (original: {original_filename})")
        
        parser = get_document_parser()
        parsed = parser.parse(file_path)
        
        if parsed is None:
            logger.error(f"[Worker] Failed to parse: {file_path}")
            raise ValueError(f"Failed to parse document: {file_path}")
        
        logger.info(f"[Worker] Parsed {parsed.file_type}: {parsed.original_filename} ({len(parsed.text)} chars, {parsed.page_count} pages)")
        
        logger.info(f"[Worker] Getting RAG instance...")
        rag = get_rag()
        ingester = DocumentIngester(rag)
        logger.info(f"[Worker] RAG ready (collection: {rag.collection_name})")
        
        processed_text = ingester._preprocess_text(parsed.text)
        logger.info(f"[Worker] Text preprocessed ({len(processed_text)} chars)")
        
        from core.config import get_config
        config = get_config()
        chunks = ingester._chunk_text(
            processed_text, 
            max_chunk_size=config.library_chunk_size,
            overlap=config.library_chunk_overlap
        )
        logger.info(f"[Worker] Created {len(chunks)} chunks")
        
        logger.info(f"[Worker] Adding {len(chunks)} chunks to Qdrant...")
        metadata = {
            "blob_id": blob_id,
            "original_filename": original_filename
        }
        count = rag.add_documents(chunks, metadata=metadata)
        logger.info(f"[Worker] Indexed {count} chunks to Qdrant")
        
        logger.info(f"[Worker] COMPLETE: {blob_id} -> {count} chunks indexed")
        
        return {
            "blob_id": blob_id,
            "chunks_indexed": count,
            "file_type": parsed.file_type,
            "original_filename": parsed.original_filename,
            "page_count": parsed.page_count
        }
    except Exception as e:
        logger.error(f"[Worker] ERROR processing {blob_id}: {e}")
        logger.error(f"[Worker] Traceback:\n{traceback.format_exc()}")
        raise


def _get_worker_settings():
    from core.config import get_config
    config = get_config()
    return {
        "host": config.redis_host,
        "port": config.redis_port,
        "timeout": config.worker_job_timeout
    }

class WorkerSettings:
    """arq worker configuration"""
    functions = [process_document]
    
    from core.config import get_config
    _config = get_config()
    redis_settings = RedisSettings(host=_config.redis_host, port=_config.redis_port)
    max_jobs = 10
    job_timeout = _config.worker_job_timeout
