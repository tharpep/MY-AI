"""Hybrid KB retrieval: pgvector cosine + PostgreSQL FTS → RRF → Voyage rerank."""

import logging
import time
from dataclasses import dataclass
from typing import Optional

from core.config import get_config
from core.database import get_pool
from rag.embedder import embed_query
from rag.reranker import rerank

logger = logging.getLogger(__name__)

RRF_K = 60  # Standard RRF constant — lower values favour top results more


@dataclass
class Chunk:
    id: str
    content: str
    filename: str
    drive_file_id: str
    chunk_index: int
    source_category: str = ""
    dense_score: float = 0.0
    fts_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0


async def _dense_search(
    conn, embedding: list[float], limit: int, categories: Optional[list[str]] = None
) -> list[Chunk]:
    """Top-limit chunks by cosine similarity (pgvector HNSW)."""
    emb_str = f"[{','.join(str(x) for x in embedding)}]"
    if categories:
        rows = await conn.fetch(
            """
            SELECT id::text, content, filename, drive_file_id, chunk_index, source_category,
                   1 - (embedding <=> $1::vector) AS score
            FROM kb_chunks
            WHERE source_category = ANY($3::text[])
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            emb_str,
            limit,
            categories,
        )
    else:
        rows = await conn.fetch(
            """
            SELECT id::text, content, filename, drive_file_id, chunk_index, source_category,
                   1 - (embedding <=> $1::vector) AS score
            FROM kb_chunks
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            emb_str,
            limit,
        )
    return [
        Chunk(
            id=r["id"],
            content=r["content"],
            filename=r["filename"] or "",
            drive_file_id=r["drive_file_id"] or "",
            chunk_index=r["chunk_index"] or 0,
            source_category=r["source_category"] or "",
            dense_score=float(r["score"]),
        )
        for r in rows
    ]


async def _fts_search(
    conn, query_text: str, limit: int, categories: Optional[list[str]] = None
) -> list[Chunk]:
    """Top-limit chunks by PostgreSQL FTS rank (GIN index)."""
    if categories:
        rows = await conn.fetch(
            """
            SELECT id::text, content, filename, drive_file_id, chunk_index, source_category,
                   ts_rank(fts, plainto_tsquery('english', $1)) AS score
            FROM kb_chunks
            WHERE fts @@ plainto_tsquery('english', $1)
              AND source_category = ANY($3::text[])
            ORDER BY score DESC
            LIMIT $2
            """,
            query_text,
            limit,
            categories,
        )
    else:
        rows = await conn.fetch(
            """
            SELECT id::text, content, filename, drive_file_id, chunk_index, source_category,
                   ts_rank(fts, plainto_tsquery('english', $1)) AS score
            FROM kb_chunks
            WHERE fts @@ plainto_tsquery('english', $1)
            ORDER BY score DESC
            LIMIT $2
            """,
            query_text,
            limit,
        )
    return [
        Chunk(
            id=r["id"],
            content=r["content"],
            filename=r["filename"] or "",
            drive_file_id=r["drive_file_id"] or "",
            chunk_index=r["chunk_index"] or 0,
            source_category=r["source_category"] or "",
            fts_score=float(r["score"]),
        )
        for r in rows
    ]


def _rrf_fuse(dense: list[Chunk], sparse: list[Chunk], limit: int) -> list[Chunk]:
    """Reciprocal Rank Fusion — combines ranked lists without score normalisation."""
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}

    for rank, chunk in enumerate(dense):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (RRF_K + rank + 1)
        by_id[chunk.id] = chunk

    for rank, chunk in enumerate(sparse):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (RRF_K + rank + 1)
        if chunk.id not in by_id:
            by_id[chunk.id] = chunk
        else:
            by_id[chunk.id].fts_score = chunk.fts_score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    result = []
    for chunk_id, rrf_score in ranked:
        c = by_id[chunk_id]
        c.rrf_score = rrf_score
        result.append(c)
    return result


async def retrieve(
    query: str,
    top_k: Optional[int] = None,
    candidates: Optional[int] = None,
    threshold: Optional[float] = None,
    categories: Optional[list[str]] = None,
) -> list[Chunk]:
    """Hybrid KB retrieval: dense + FTS → RRF → Voyage rerank.

    Args:
        query:      User query text.
        top_k:      Final chunks to return (default: config.chat_kb_top_k).
        candidates: Candidates fetched before reranking (default: config.rerank_candidates).
        threshold:  Min rerank_score to keep (default: config.chat_kb_similarity_threshold).
        categories: Optional list of source_category values to restrict search to.

    Returns:
        Chunks ordered by relevance, length <= top_k.
    """
    config = get_config()
    top_k = top_k or config.chat_kb_top_k
    candidates = candidates or config.rerank_candidates
    threshold = threshold if threshold is not None else config.chat_kb_similarity_threshold

    pool = get_pool()

    logger.debug(
        f"retrieve: query='{query[:80]}' top_k={top_k} candidates={candidates} "
        f"threshold={threshold} categories={categories}"
    )

    # 1. Embed the query
    t0 = time.perf_counter()
    embedding = await embed_query(query)
    logger.debug(f"  [1] embed_query: {time.perf_counter() - t0:.3f}s, dim={len(embedding)}")

    async with pool.acquire() as conn:
        async with conn.transaction():
            # SET LOCAL only takes effect for the duration of this transaction, so it
            # never leaks the setting onto a pooled connection reused by another request.
            # hnsw.ef_search doesn't accept a bind parameter — config.hnsw_ef_search is
            # a pydantic-validated (ge=10, le=1000) int from our own settings, not user
            # input, so inlining it is safe.
            await conn.execute(f"SET LOCAL hnsw.ef_search = {config.hnsw_ef_search}")

            # 2. Dense search
            t0 = time.perf_counter()
            dense = await _dense_search(conn, embedding, candidates, categories)
            logger.debug(
                f"  [2] dense search: {len(dense)} candidates in {time.perf_counter() - t0:.3f}s"
                + (f", top score={dense[0].dense_score:.4f}" if dense else "")
            )

            # 3. FTS search (skipped when sparse_weight == 0 → dense-only mode)
            sparse: list[Chunk] = []
            if config.hybrid_sparse_weight > 0:
                t0 = time.perf_counter()
                sparse = await _fts_search(conn, query, candidates, categories)
                logger.debug(
                    f"  [3] fts search: {len(sparse)} candidates in {time.perf_counter() - t0:.3f}s"
                    + (f", top score={sparse[0].fts_score:.4f}" if sparse else "")
                )
            else:
                logger.debug("  [3] fts search: skipped (sparse_weight=0)")

    # 4. RRF fusion (or pass-through if no sparse results)
    fused = _rrf_fuse(dense, sparse, candidates) if sparse else dense[:candidates]
    logger.debug(
        f"  [4] rrf fusion: {len(fused)} candidates"
        + (f", top rrf_score={fused[0].rrf_score:.4f}" if fused else "")
    )

    if not fused:
        return []

    # 5. Rerank with Voyage rerank-2.5
    if config.rerank_enabled:
        t0 = time.perf_counter()
        fused = await rerank(query, fused, top_k)
        logger.debug(
            f"  [5] rerank: {len(fused)} chunks in {time.perf_counter() - t0:.3f}s"
            + (f", top rerank_score={fused[0].rerank_score:.4f}" if fused else "")
        )
    else:
        fused = fused[:top_k]
        logger.debug(f"  [5] rerank: skipped, truncated to {len(fused)} chunks")

    # 6. Apply similarity threshold (only meaningful when reranking is enabled;
    #    rerank_score stays 0.0 when reranking is skipped, so don't filter then)
    before = len(fused)
    if threshold > 0 and config.rerank_enabled:
        fused = [c for c in fused if c.rerank_score >= threshold]
    logger.debug(f"  [6] threshold ({threshold}): {before} → {len(fused)} chunks returned")

    return fused
