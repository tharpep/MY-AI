"""KB sync engine — Drive → kb_chunks with kb_sources change tracking."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from core.config import get_config
from core.database import get_pool
from llm.gateway import AIGateway
from rag.chunking import chunk_markdown, chunk_text
from rag.embedder import embed_documents
from rag.loader import DriveFileRecord, download_file, list_drive_files, parse_content

logger = logging.getLogger(__name__)

_SUMMARY_UNAVAILABLE = "[unavailable]"


def _generate_summary(text: str, gw: AIGateway) -> str:
    """Call Haiku via the AI gateway to produce a 1-2 sentence document summary."""
    try:
        return gw.chat(
            f"In 1-2 sentences, describe what this document is about and what kind of "
            f"information it contains. Be specific about names, projects, or topics if evident. "
            f"Reply with only the description, no preamble.\n\n{text[:2000]}"
        )
    except Exception as exc:
        logger.warning(f"Summary generation failed: {exc}")
        return _SUMMARY_UNAVAILABLE


def _contextualize_chunk(chunk: str, filename: str, summary: str, section_title: str = "") -> str:
    """Prepend a short doc-aware (and section-aware, if known) context blurb before embedding.

    Implements Anthropic's Contextual Retrieval. The contextualized text is what
    gets embedded AND stored in `content` — so the generated `fts` tsvector also
    benefits — the deliberate tradeoff (per issue #102's "simplest first cut") is
    that retrieved chunks show this prefix.
    """
    context_bits = [f"From '{filename}'"]
    if summary and summary != _SUMMARY_UNAVAILABLE:
        context_bits.append(f"({summary})")
    if section_title:
        context_bits.append(f"[{section_title}]")
    prefix = " ".join(context_bits) + ": "
    return prefix + chunk


# Stay well under Voyage's 128-input / 320K-token per-request limit
_EMBED_BATCH = 96


async def _get_all_kb_sources(pool) -> dict[str, dict]:
    """Fetch all kb_sources rows keyed by file_id."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT file_id, filename, category, modified_time, last_synced, chunk_count, summary, status "
            "FROM kb_sources"
        )
    return {r["file_id"]: dict(r) for r in rows}


async def _upsert_kb_source(
    conn,
    file_id: str,
    filename: str,
    category: str,
    modified_time: str,
    chunk_count: int,
    summary: str = "",
    raw_content: str = "",
    origin: str = "drive",
) -> None:
    """Insert or update a kb_sources record, setting last_synced to now."""
    modified_dt = datetime.fromisoformat(modified_time.replace("Z", "+00:00"))
    await conn.execute(
        """
        INSERT INTO kb_sources
            (file_id, filename, category, modified_time, last_synced, chunk_count, summary,
             status, raw_content, origin)
        VALUES ($1, $2, $3, $4, NOW(), $5, $6, 'active', $7, $8)
        ON CONFLICT (file_id) DO UPDATE SET
            filename      = EXCLUDED.filename,
            category      = EXCLUDED.category,
            modified_time = EXCLUDED.modified_time,
            last_synced   = NOW(),
            chunk_count   = EXCLUDED.chunk_count,
            summary       = EXCLUDED.summary,
            status        = 'active',
            raw_content   = EXCLUDED.raw_content,
            origin        = EXCLUDED.origin
        """,
        file_id,
        filename,
        category,
        modified_dt,
        chunk_count,
        summary,
        raw_content,
        origin,
    )


async def _mark_source_error(pool, file: DriveFileRecord) -> None:
    """Record that a sync attempt for this file failed.

    Without this, a file that starts failing (corrupt re-upload, parse
    exception, etc.) after having synced successfully once keeps whatever
    status it last had — 'active' — forever, so it looks healthy in
    GET /kb/sources while silently retrying and failing on every sync. Creates
    a minimal row (status='error') for a file that has never synced at all,
    or flips an existing row's status without touching its other columns.
    """
    modified_dt = datetime.fromisoformat(file.modified_time.replace("Z", "+00:00"))
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO kb_sources (file_id, filename, category, modified_time, last_synced, status, origin)
            VALUES ($1, $2, $3, $4, NOW(), 'error', 'drive')
            ON CONFLICT (file_id) DO UPDATE SET
                status      = 'error',
                last_synced = NOW()
            """,
            file.id,
            file.name,
            file.category,
            modified_dt,
        )


async def _upsert_file_chunks(
    pool,
    drive_file_id: str,
    filename: str,
    source_category: str,
    chunks: list[str],
    embeddings: list[list[float]],
    section_titles: list[str] | None = None,
) -> int:
    """Delete existing chunks for this file and insert new ones atomically.

    Returns the number of chunks inserted.
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM kb_chunks WHERE drive_file_id = $1", drive_file_id
            )
            if not chunks:
                return 0
            await conn.executemany(
                """
                INSERT INTO kb_chunks
                    (content, embedding, source_category, drive_file_id, filename, chunk_index, metadata)
                VALUES ($1, $2::vector, $3, $4, $5, $6, $7::jsonb)
                """,
                [
                    (
                        chunk,
                        f"[{','.join(str(x) for x in emb)}]",
                        source_category,
                        drive_file_id,
                        filename,
                        idx,
                        json.dumps(
                            {"section_title": section_titles[idx]}
                            if section_titles and section_titles[idx]
                            else {}
                        ),
                    )
                    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
                ],
            )
    return len(chunks)


async def _remove_deleted_files(pool, file_ids: list[str]) -> None:
    """Delete chunks and mark kb_sources as deleted for files no longer in Drive."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            for fid in file_ids:
                await conn.execute(
                    "DELETE FROM kb_chunks WHERE drive_file_id = $1", fid
                )
                await conn.execute(
                    "UPDATE kb_sources SET status = 'deleted', last_synced = NOW() "
                    "WHERE file_id = $1",
                    fid,
                )


def _needs_sync(file: DriveFileRecord, source: Optional[dict]) -> bool:
    """Return True if the file is new, has been modified since last sync, or has no summary."""
    if source is None:
        return True
    if not source.get("summary"):
        return True
    last_synced: Optional[datetime] = source.get("last_synced")
    if last_synced is None:
        return True
    file_modified = datetime.fromisoformat(file.modified_time.replace("Z", "+00:00"))
    # last_synced from asyncpg is already timezone-aware
    if last_synced.tzinfo is None:
        last_synced = last_synced.replace(tzinfo=timezone.utc)
    return file_modified > last_synced


async def _sync_one_file(
    file: DriveFileRecord,
    gw: AIGateway,
    pool,
    config,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Download, parse, summarize, chunk, embed, and upsert a single file.

    Returns {"status": "synced", "chunks": N} | {"status": "empty"} | {"status": "error", "error": str}.
    Bounded by `semaphore` so at most `kb_sync_concurrency` files are in flight at once —
    Drive downloads, Haiku summaries, and Voyage embeddings all happen per-file, so
    unbounded concurrency would hammer all three at once.
    """
    async with semaphore:
        try:
            data, content_type, _ = await download_file(file.id)
            logger.debug(f"  downloaded '{file.name}': {len(data):,} bytes, type={content_type}")

            text = parse_content(data, content_type, file.name)
            logger.debug(f"  parsed '{file.name}': {len(text):,} chars")

            if not text.strip():
                logger.warning(f"No text extracted from '{file.name}', skipping")
                return {"status": "empty"}

            summary = await asyncio.to_thread(_generate_summary, text, gw)
            logger.debug(f"  summary '{file.name}': {summary[:80]!r}")

            is_markdown = file.name.lower().endswith((".md", ".markdown"))
            section_titles: list[str] | None = None
            if is_markdown:
                raw_chunks = chunk_markdown(
                    text, chunk_size=config.kb_chunk_size, overlap=config.kb_chunk_overlap
                )
                chunks = [
                    _contextualize_chunk(chunk, file.name, summary, section_title)
                    for chunk, section_title in raw_chunks
                ]
                # Stored in kb_chunks.metadata so a caller can cite "section X" instead
                # of only ever pointing at the whole document.
                section_titles = [section_title for _, section_title in raw_chunks]
            else:
                raw_chunks = chunk_text(
                    text, chunk_size=config.kb_chunk_size, overlap=config.kb_chunk_overlap
                )
                chunks = [_contextualize_chunk(chunk, file.name, summary) for chunk in raw_chunks]
            if not chunks:
                return {"status": "empty"}

            n_batches = (len(chunks) + _EMBED_BATCH - 1) // _EMBED_BATCH
            logger.debug(
                f"  embedding '{file.name}': {len(chunks)} chunk(s) in {n_batches} batch(es)"
            )
            all_embeddings: list[list[float]] = []
            for i in range(0, len(chunks), _EMBED_BATCH):
                batch = chunks[i : i + _EMBED_BATCH]
                embs = await embed_documents(batch)
                all_embeddings.extend(embs)

            inserted = await _upsert_file_chunks(
                pool,
                drive_file_id=file.id,
                filename=file.name,
                source_category=file.category,
                chunks=chunks,
                embeddings=all_embeddings,
                section_titles=section_titles,
            )

            # Update kb_sources within its own connection (outside chunk transaction)
            async with pool.acquire() as conn:
                await _upsert_kb_source(
                    conn,
                    file.id,
                    file.name,
                    file.category,
                    file.modified_time,
                    inserted,
                    summary,
                    raw_content=text,
                    origin="drive",
                )

            logger.info(f"Synced '{file.name}': {inserted} chunk(s)")
            return {"status": "synced", "chunks": inserted}

        except Exception as e:
            logger.exception(f"Error syncing '{file.name}'")
            error_msg = f"{file.name}: {e}"
            try:
                await _mark_source_error(pool, file)
            except Exception:
                # Don't let a failure to record the failure mask the original error.
                logger.exception(f"Failed to mark kb_sources status=error for '{file.name}'")
            return {"status": "error", "error": error_msg}


# Arbitrary constant identifying this app's sync lock in Postgres's advisory-lock
# keyspace (a 64-bit int shared cluster-wide) — has no meaning beyond being a
# fixed, unique key both callers agree on.
_SYNC_ADVISORY_LOCK_KEY = 837462910


async def sync_drive(force: bool = False) -> dict:
    """Sync all KB Drive subfolders, serialized by a Postgres advisory lock.

    A manual POST /kb/sync racing a cron-triggered sync (or two manual
    triggers) previously ran two full sync passes concurrently — with nothing
    stopping their DELETE+INSERT transactions on the same file from
    interleaving, that could duplicate a file's chunks. The lock is held on
    one dedicated connection for the whole sync; a caller that can't acquire
    it returns immediately instead of doing redundant work.
    """
    pool = get_pool()
    conn = await pool.acquire()
    try:
        acquired = await conn.fetchval(
            "SELECT pg_try_advisory_lock($1)", _SYNC_ADVISORY_LOCK_KEY
        )
        if not acquired:
            logger.warning("sync_drive() skipped — another sync is already in progress")
            return {
                "files_synced": 0,
                "files_skipped": 0,
                "files_deleted": 0,
                "chunks_inserted": 0,
                "errors": ["Sync already in progress — skipped."],
                "synced_at": datetime.now(timezone.utc).isoformat(),
            }
        return await _sync_drive_locked(force)
    finally:
        try:
            await conn.execute("SELECT pg_advisory_unlock($1)", _SYNC_ADVISORY_LOCK_KEY)
        finally:
            await pool.release(conn)


async def _sync_drive_locked(force: bool = False) -> dict:
    """The actual sync pass — only ever called while sync_drive() holds the lock.

    Files needing sync are processed concurrently (bounded by kb_sync_concurrency) rather
    than one at a time — each file's download/summarize/embed/upsert pipeline is otherwise
    independent, and sequential processing was the bottleneck on large Drive folders.

    Args:
        force: If True, re-sync every file regardless of modification time.

    Returns:
        dict with keys: files_synced, files_skipped, files_deleted, chunks_inserted,
                        errors, synced_at
    """
    config = get_config()
    pool = get_pool()
    gw = AIGateway()

    # All files across all KB subfolders (category comes from each file's DriveFileRecord)
    drive_files = await list_drive_files()
    logger.info(f"Drive sync: {len(drive_files)} file(s) found across all KB subfolders")

    # Existing kb_sources state for change detection and deletion tracking
    existing_sources = await _get_all_kb_sources(pool)
    drive_ids = {f.id for f in drive_files}

    # Files that no longer exist in Drive but are still active in kb_sources
    deleted_ids = [
        fid
        for fid, src in existing_sources.items()
        if src.get("status") == "active" and fid not in drive_ids
    ]

    # Remove deleted files first
    if deleted_ids:
        await _remove_deleted_files(pool, deleted_ids)
        logger.info(f"Removed {len(deleted_ids)} deleted file(s) from KB")

    files_to_sync = [
        f for f in drive_files if force or _needs_sync(f, existing_sources.get(f.id))
    ]
    files_skipped = len(drive_files) - len(files_to_sync)
    if files_skipped:
        logger.debug(f"Skipping {files_skipped} file(s) — not modified since last sync")

    semaphore = asyncio.Semaphore(config.kb_sync_concurrency)
    results = await asyncio.gather(
        *[_sync_one_file(f, gw, pool, config, semaphore) for f in files_to_sync]
    )

    files_synced = sum(1 for r in results if r["status"] == "synced")
    chunks_inserted = sum(r.get("chunks", 0) for r in results if r["status"] == "synced")
    errors = [r["error"] for r in results if r["status"] == "error"]

    return {
        "files_synced": files_synced,
        "files_skipped": files_skipped,
        "files_deleted": len(deleted_ids),
        "chunks_inserted": chunks_inserted,
        "errors": errors,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }
