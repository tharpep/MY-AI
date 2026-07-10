"""KB sync routes — Drive → kb_chunks sync management, plus direct (non-Drive) ingest."""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from core.config import get_config
from core.database import get_pool
from llm.gateway import AIGateway
from rag.chunking import chunk_text
from rag.embedder import embed_documents
from rag.sync import (
    _EMBED_BATCH,
    _generate_summary,
    _upsert_file_chunks,
    _upsert_kb_source,
    sync_drive,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class SyncResponse(BaseModel):
    files_synced: int
    files_skipped: int
    files_deleted: int
    chunks_inserted: int
    errors: list[str]
    synced_at: str


@router.post("/kb/sync", response_model=SyncResponse)
async def run_sync(
    force: bool = Query(
        default=False,
        description="Re-sync all files regardless of modification time",
    ),
):
    """Sync KB Drive folder into kb_chunks.

    By default only processes new or modified files (smart incremental sync
    using kb_sources change tracking). Pass force=true to re-sync everything.
    """
    try:
        result = await sync_drive(force=force)  # category comes from each file's Drive folder
        return SyncResponse(**result)
    except Exception as e:
        logger.error(f"KB sync failed: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/kb/sources")
async def list_kb_sources():
    """List all files tracked in kb_sources with their sync status."""
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT file_id, filename, category, modified_time, last_synced,
                       chunk_count, status
                FROM kb_sources
                ORDER BY filename
                """
            )
        return {
            "sources": [
                {
                    "file_id": r["file_id"],
                    "filename": r["filename"],
                    "category": r["category"],
                    "modified_time": r["modified_time"].isoformat() if r["modified_time"] else None,
                    "last_synced": r["last_synced"].isoformat() if r["last_synced"] else None,
                    "chunk_count": r["chunk_count"],
                    "status": r["status"],
                }
                for r in rows
            ],
            "count": len(rows),
        }
    except Exception as e:
        logger.error(f"Failed to list KB sources: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/kb/files")
async def list_kb_files():
    """List all files currently indexed in kb_chunks, with chunk counts and category."""
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT drive_file_id, filename, source_category, COUNT(*) AS chunk_count
                FROM kb_chunks
                GROUP BY drive_file_id, filename, source_category
                ORDER BY filename
                """
            )
        return {
            "files": [
                {
                    "drive_file_id": r["drive_file_id"],
                    "filename": r["filename"],
                    "source_category": r["source_category"],
                    "chunk_count": r["chunk_count"],
                }
                for r in rows
            ],
            "count": len(rows),
        }
    except Exception as e:
        logger.error(f"Failed to list KB files: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/kb", status_code=204)
async def clear_kb():
    """Truncate kb_chunks and kb_sources — removes all indexed content and sync state."""
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute("TRUNCATE TABLE kb_chunks")
            await conn.execute("TRUNCATE TABLE kb_sources")
    except Exception as e:
        logger.error(f"Failed to clear KB: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/kb/files/{drive_file_id}", status_code=204)
async def delete_kb_file(drive_file_id: str):
    """Remove a file's chunks from kb_chunks and mark it deleted in kb_sources."""
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM kb_chunks WHERE drive_file_id = $1", drive_file_id
            )
            deleted_count = int(result.split()[-1])
            if deleted_count == 0:
                raise HTTPException(
                    status.HTTP_404_NOT_FOUND,
                    detail=f"No chunks found for drive_file_id={drive_file_id}",
                )
            await conn.execute(
                "UPDATE kb_sources SET status = 'deleted', last_synced = NOW() "
                "WHERE file_id = $1",
                drive_file_id,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete KB file {drive_file_id}: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


class IngestTextRequest(BaseModel):
    title: str
    content: str
    category: str | None = None


class IngestUrlRequest(BaseModel):
    url: str
    category: str | None = None


class IngestResponse(BaseModel):
    file_id: str
    filename: str
    category: str | None
    origin: str
    chunk_count: int
    summary: str


async def _ingest_text(file_id: str, filename: str, category: str | None, content: str, origin: str) -> IngestResponse:
    """Shared pipeline for direct (non-Drive) ingest: summarize, chunk, embed, upsert.

    Reuses the exact per-file upsert helpers the Drive sync path uses, so a
    directly-ingested doc is indistinguishable downstream from a synced one
    except for its `origin`.
    """
    config = get_config()
    gw = AIGateway()

    if not content.strip():
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="No content to ingest")

    summary = await asyncio.to_thread(_generate_summary, content, gw)

    chunks = chunk_text(content, chunk_size=config.kb_chunk_size, overlap=config.kb_chunk_overlap)
    if not chunks:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Content produced no chunks")

    all_embeddings: list[list[float]] = []
    for i in range(0, len(chunks), _EMBED_BATCH):
        batch = chunks[i : i + _EMBED_BATCH]
        all_embeddings.extend(await embed_documents(batch))

    pool = get_pool()
    inserted = await _upsert_file_chunks(
        pool,
        drive_file_id=file_id,
        filename=filename,
        source_category=category or "",
        chunks=chunks,
        embeddings=all_embeddings,
    )

    modified_time = datetime.now(timezone.utc).isoformat()
    async with pool.acquire() as conn:
        await _upsert_kb_source(
            conn,
            file_id,
            filename,
            category or "",
            modified_time,
            inserted,
            summary,
            raw_content=content,
            origin=origin,
        )

    return IngestResponse(
        file_id=file_id,
        filename=filename,
        category=category,
        origin=origin,
        chunk_count=inserted,
        summary=summary,
    )


@router.post("/kb/ingest/text", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_text(body: IngestTextRequest):
    """Directly ingest arbitrary text content — no Drive file required."""
    file_id = f"text:{uuid.uuid4()}"
    try:
        return await _ingest_text(file_id, body.title, body.category, body.content, origin="text")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text ingest failed: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/kb/ingest/url", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_url(body: IngestUrlRequest):
    """Fetch a URL via the gateway's web-fetch (Tavily extract) and ingest its text."""
    config = get_config()
    file_id = f"url:{hashlib.sha256(body.url.encode()).hexdigest()}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{config.api_gateway_url.rstrip('/')}/search/web/fetch",
            json={"url": body.url},
            headers={"X-API-Key": config.api_gateway_key},
        )
    if resp.status_code != 200:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=f"URL fetch failed: {resp.text}")

    data = resp.json()
    results = data.get("results") or []
    if not results or not results[0].get("raw_content"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"No content extracted from {body.url}")
    content = results[0]["raw_content"]

    try:
        return await _ingest_text(file_id, body.url, body.category, content, origin="url")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL ingest failed: {e}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
