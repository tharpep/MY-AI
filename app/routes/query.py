"""KB query routes — search kb_chunks via hybrid retrieval."""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from core.database import get_pool
from rag.query_processor import QueryProcessor
from rag.retriever import retrieve

logger = logging.getLogger(__name__)

router = APIRouter()


class KBSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    candidates: Optional[int] = None
    threshold: Optional[float] = None
    categories: Optional[list[str]] = None
    expand_query: bool = False


class KBChunkResult(BaseModel):
    content: str
    filename: str
    drive_file_id: str
    chunk_index: int
    source_category: str
    rerank_score: float
    dense_score: float
    rrf_score: float


class KBSearchResponse(BaseModel):
    results: list[KBChunkResult]
    count: int
    query: str
    expanded_query: Optional[str] = None


@router.post("/kb/search", response_model=KBSearchResponse)
async def search_kb(body: KBSearchRequest):
    """Search kb_chunks using hybrid retrieval (dense + FTS → RRF → Voyage rerank)."""
    query = body.query.strip()
    if not query:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Query cannot be empty")

    try:
        expanded_query = None
        logger.debug(
            f"kb/search: query='{query[:80]}' top_k={body.top_k} candidates={body.candidates} "
            f"threshold={body.threshold} categories={body.categories} expand={body.expand_query}"
        )

        if body.expand_query:
            qp = QueryProcessor()
            expanded_query = await asyncio.to_thread(qp.expand, query)
            logger.debug(f"kb/search: expanded query='{expanded_query[:80]}'")
            query = expanded_query

        chunks = await retrieve(
            query=query,
            top_k=body.top_k,
            candidates=body.candidates,
            threshold=body.threshold,
            categories=body.categories or None,
        )

        logger.debug(
            f"kb/search: returning {len(chunks)} chunk(s)"
            + (f", top rerank_score={chunks[0].rerank_score:.4f}" if chunks else "")
        )

        return KBSearchResponse(
            results=[
                KBChunkResult(
                    content=c.content,
                    filename=c.filename,
                    drive_file_id=c.drive_file_id,
                    chunk_index=c.chunk_index,
                    source_category=c.source_category,
                    rerank_score=c.rerank_score,
                    dense_score=c.dense_score,
                    rrf_score=c.rrf_score,
                )
                for c in chunks
            ],
            count=len(chunks),
            query=body.query,
            expanded_query=expanded_query,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("KB search failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


class KBIndexEntry(BaseModel):
    file_id: str
    filename: str
    category: str
    chunk_count: int
    summary: str | None


class KBIndexResponse(BaseModel):
    index: list[KBIndexEntry]
    count: int


@router.get("/kb/index", response_model=KBIndexResponse)
async def kb_index():
    """Return all active KB sources with summaries — a lightweight document directory."""
    try:
        pool = get_pool()
        rows = await pool.fetch(
            """
            SELECT file_id, filename, category, chunk_count, summary
            FROM kb_sources
            WHERE status = 'active'
            ORDER BY category, filename
            """
        )
        entries = [
            KBIndexEntry(
                file_id=r["file_id"],
                filename=r["filename"],
                category=r["category"],
                chunk_count=r["chunk_count"],
                summary=r["summary"],
            )
            for r in rows
        ]
        return KBIndexResponse(index=entries, count=len(entries))
    except Exception as e:
        logger.exception("Failed to get KB index")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/kb/stats")
async def kb_stats():
    """Return total chunk count and distinct file count from kb_chunks."""
    try:
        pool = get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT COUNT(*) AS chunk_count,
                       COUNT(DISTINCT drive_file_id) AS file_count
                FROM kb_chunks
                """
            )
        return {
            "chunk_count": row["chunk_count"],
            "file_count": row["file_count"],
        }
    except Exception as e:
        logger.exception("Failed to get KB stats")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
