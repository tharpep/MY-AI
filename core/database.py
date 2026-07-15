"""PostgreSQL connection pool and schema initialization."""

import logging
from typing import Optional

import asyncpg

from core.config import get_config

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None

_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS kb_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content         TEXT NOT NULL,
    embedding       vector(1024),
    fts             TSVECTOR GENERATED ALWAYS AS
                        (to_tsvector('english', content)) STORED,
    source_category TEXT,
    drive_file_id   TEXT,
    filename        TEXT,
    chunk_index     INTEGER,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS kb_chunks_embedding_hnsw_idx
    ON kb_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 24, ef_construction = 100);

CREATE INDEX IF NOT EXISTS kb_chunks_fts_idx
    ON kb_chunks USING gin (fts);

CREATE INDEX IF NOT EXISTS kb_chunks_drive_file_id_idx
    ON kb_chunks (drive_file_id);

CREATE INDEX IF NOT EXISTS kb_chunks_source_category_idx
    ON kb_chunks (source_category);

CREATE TABLE IF NOT EXISTS kb_sources (
    file_id       TEXT PRIMARY KEY,
    filename      TEXT,
    category      TEXT,
    modified_time TIMESTAMPTZ,
    last_synced   TIMESTAMPTZ,
    chunk_count   INT DEFAULT 0,
    summary       TEXT,
    status        TEXT DEFAULT 'active'
);

CREATE INDEX IF NOT EXISTS kb_sources_category_idx
    ON kb_sources (category);

CREATE INDEX IF NOT EXISTS kb_sources_status_idx
    ON kb_sources (status);
"""

# Migrations applied to existing tables on startup.
# Safe to run repeatedly — all are idempotent.
_MIGRATION_SQL = """
ALTER TABLE kb_chunks ADD COLUMN IF NOT EXISTS source_category TEXT;
ALTER TABLE kb_chunks DROP COLUMN IF EXISTS folder;
ALTER TABLE kb_sources ADD COLUMN IF NOT EXISTS summary TEXT;
ALTER TABLE kb_sources ADD COLUMN IF NOT EXISTS raw_content TEXT;
ALTER TABLE kb_sources ADD COLUMN IF NOT EXISTS origin TEXT DEFAULT 'drive';
DROP INDEX IF EXISTS kb_chunks_embedding_idx;

-- De-duplicate any existing (drive_file_id, chunk_index) collisions — possible
-- today because nothing prevented two overlapping sync_drive() calls (a manual
-- /kb/sync racing a cron-triggered one) from both inserting for the same file —
-- before adding the uniqueness constraint below. Keeps the most recently
-- created row of each colliding pair; (created_at, id) breaks exact-timestamp
-- ties deterministically.
DELETE FROM kb_chunks a USING kb_chunks b
WHERE a.drive_file_id IS NOT NULL
  AND a.drive_file_id = b.drive_file_id
  AND a.chunk_index = b.chunk_index
  AND (a.created_at, a.id) < (b.created_at, b.id);

-- Partial (not full) uniqueness: chunks from ingest_text/ingest_url leave
-- drive_file_id NULL, and NULL <> NULL under UNIQUE anyway, but being
-- explicit documents the intent — this only guards actual Drive-synced files.
CREATE UNIQUE INDEX IF NOT EXISTS kb_chunks_drive_file_chunk_idx
    ON kb_chunks (drive_file_id, chunk_index) WHERE drive_file_id IS NOT NULL;
"""


async def init_pool() -> None:
    """Create the asyncpg pool and initialize the schema."""
    global _pool

    config = get_config()
    if not config.database_url:
        logger.warning("DATABASE_URL not set — skipping database init")
        return

    # asyncpg expects postgresql://, not postgresql+asyncpg://
    dsn = config.database_url.replace("postgresql+asyncpg://", "postgresql://")

    logger.info("Connecting to PostgreSQL...")
    _pool = await asyncpg.create_pool(dsn=dsn, min_size=2, max_size=10)

    async with _pool.acquire() as conn:
        await conn.execute(_SCHEMA_SQL)
        await conn.execute(_MIGRATION_SQL)

    logger.info("Database pool ready and schema initialized")


def get_pool() -> asyncpg.Pool:
    """Return the active pool. Raises if not initialized."""
    if _pool is None:
        raise RuntimeError("Database pool not initialized — call init_pool() first")
    return _pool


async def close_pool() -> None:
    """Close the pool on shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("Database pool closed")
