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
    Pipeline: Get Blob -> Parse -> Preprocess -> Chunk -> Embed -> Store
    
    Args:
        ctx: arq context (contains redis connection)
        blob_id: The blob to process
        
    Returns:
        dict with processing results
    """
    import traceback
    
    try:
        from core.file_storage import get_blob_storage
        from rag.document_parser import get_document_parser
        from rag.document_ingester import DocumentIngester
        from rag.rag_setup import get_rag
        
        logger.info(f"[Worker] Starting processing for blob: {blob_id}")
        
        # Step 1: Get blob from storage
        storage = get_blob_storage()
        file_path = storage.get(blob_id)
        blob_info = storage.get_info(blob_id)  # Get original filename from manifest
        
        if file_path is None:
            logger.error(f"[Worker] Blob not found: {blob_id}")
            raise ValueError(f"Blob not found: {blob_id}")
        
        original_filename = blob_info.original_filename if blob_info else file_path.name
        logger.info(f"[Worker] Found blob at: {file_path} (original: {original_filename})")
        
        # Step 2: Parse document (extract text)
        parser = get_document_parser()
        parsed = parser.parse(file_path)
        
        if parsed is None:
            logger.error(f"[Worker] Failed to parse: {file_path}")
            raise ValueError(f"Failed to parse document: {file_path}")
        
        logger.info(f"[Worker] Parsed {parsed.file_type}: {parsed.original_filename} ({len(parsed.text)} chars, {parsed.page_count} pages)")
        
        # Step 3: Get RAG instance (singleton) and create ingester
        logger.info(f"[Worker] Getting RAG instance...")
        rag = get_rag()
        ingester = DocumentIngester(rag)
        logger.info(f"[Worker] RAG ready (collection: {rag.collection_name})")
        
        # Step 4: Preprocess text
        processed_text = ingester._preprocess_text(parsed.text)
        logger.info(f"[Worker] Text preprocessed ({len(processed_text)} chars)")
        
        # Step 5: Chunk text
        chunks = ingester._chunk_text(processed_text, max_chunk_size=1000)
        logger.info(f"[Worker] Created {len(chunks)} chunks")
        
        # Step 6: Add to vector database (Qdrant) with blob tracking metadata
        logger.info(f"[Worker] Adding {len(chunks)} chunks to Qdrant...")
        metadata = {
            "blob_id": blob_id,
            "original_filename": original_filename  # From blob manifest, not parsed path
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


# arq worker settings - used when running: arq rag.workers.WorkerSettings
class WorkerSettings:
    """arq worker configuration"""
    functions = [process_document]
    redis_settings = RedisSettings(host='localhost', port=6379)
    max_jobs = 10
    job_timeout = 300  # 5 minutes per job
