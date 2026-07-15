"""PostgreSQL connection pool and schema initialization."""

import logging
from typing import Optional

import asyncpg

from core.config import get_config
from core.migrations import run_migrations

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None


async def init_pool() -> None:
    """Run pending migrations, then create the asyncpg pool."""
    global _pool

    config = get_config()
    if not config.database_url:
        logger.warning("DATABASE_URL not set — skipping database init")
        return

    # asyncpg expects postgresql://, not postgresql+asyncpg://
    dsn = config.database_url.replace("postgresql+asyncpg://", "postgresql://")

    # Run before the pool exists (see core/migrations/__init__.py) — a real
    # failure here should crash startup, same as it already does for
    # api-gateway's migrations, instead of deploying a revision that looks
    # healthy and then 500s on every DB-touching request.
    await run_migrations(dsn)

    logger.info("Connecting to PostgreSQL...")
    _pool = await asyncpg.create_pool(dsn=dsn, min_size=2, max_size=10)

    logger.info("Database pool ready")


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
