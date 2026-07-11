# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Cloud-first **knowledge base service** built with **FastAPI** (Python 3.11+). Ingests documents from Google Drive (via the api-gateway), stores vectors in PostgreSQL + pgvector (GCP Cloud SQL), and exposes a hybrid retrieval API. Part of a larger personal AI ecosystem — see `../api-gateway/developmentplan.md` for the full architecture.

Managed with **Poetry**. Deployed to **GCP Cloud Run**.

## Commands

```bash
# Install dependencies
poetry install

# Run dev server (hot reload)
poetry run uvicorn app.main:app --reload

# Lint
ruff check .

# Format
ruff format .
# or: black .
```

No test suite exists yet.

## Architecture

**Entry point:** `app/main.py` — creates FastAPI app with lifespan that initialises the asyncpg pool + schema and the `AIGateway` singleton. Mounts all routers under `/v1` (and health under `/health`).

**Configuration:** `core/config.py` — `AppConfig` via pydantic-settings, loaded from `.env`. Singleton via `get_config()`. Key env vars: `DATABASE_URL`, `VOYAGE_API_KEY`, `API_GATEWAY_URL`, `API_GATEWAY_KEY`.

### Module Layout

```
app/
  main.py            — FastAPI app + lifespan, CORS, router mounts
  routes/
    health.py        — GET /health, GET /health/detailed
    llm.py           — POST /v1/chat/completions (OpenAI-compat, optional KB injection)
                       GET  /v1/models
    query.py         — POST /v1/kb/search (hybrid retrieval), GET /v1/kb/stats
    ingest.py        — POST /v1/kb/sync (Drive sync; force=true for full re-sync)
                       GET  /v1/kb/sources, GET /v1/kb/files
                       DELETE /v1/kb, DELETE /v1/kb/files/{drive_file_id}
    config.py        — GET /v1/config, PATCH /v1/config

core/
  config.py          — AppConfig (pydantic-settings singleton)
  database.py        — asyncpg pool + schema init (kb_chunks, kb_sources, pgvector)

llm/
  gateway.py         — AIGateway: routes LLM calls to api-gateway /ai/v1/chat/completions
  base_client.py     — provider interface
  local.py           — Ollama provider (local dev only, never deployed)
  providers/
    anthropic.py     — direct Anthropic client (unused; routing goes through gateway)

rag/
  embedder.py        — Voyage AI embed_documents() / embed_query() (async, batched at 96)
  loader.py          — lists + downloads files from api-gateway /storage endpoints;
                       parses PDF (PyPDF2), DOCX (python-docx), plain text/CSV/markdown
  chunking.py        — text chunking (langchain-text-splitters)
  sync.py            — sync_drive(): Drive → kb_chunks using kb_sources for change detection
  retriever.py       — hybrid search: dense (pgvector) + FTS → RRF → Voyage rerank
  reranker.py        — Voyage rerank-2.5 via voyageai SDK
  query_processor.py — LLM query expansion via gateway (optional, per-request flag)
```

### Database Schema (`core/database.py`)

Tables auto-created on startup:

- **kb_chunks** — content, embedding (vector 1024), fts (tsvector), source_category, drive_file_id, filename, chunk_index, metadata. Indexes: HNSW (embedding), GIN (fts), btree (drive_file_id, source_category).
- **kb_sources** — file_id (PK), filename, category, modified_time, last_synced, chunk_count, status. Used for incremental sync and deletion tracking.

### Retrieval Pipeline (`rag/retriever.py`)

```
embed_query(query)
    → dense_search (pgvector HNSW cosine, top candidates)
    → fts_search (PostgreSQL plainto_tsquery, top candidates)  [skipped if sparse_weight=0]
    → RRF fusion
    → Voyage rerank-2.5 (top_k)
    → similarity threshold filter
```

Config knobs (all in `core/config.py`): `hybrid_sparse_weight`, `rerank_enabled`, `rerank_candidates`, `rerank_model`, `chat_kb_top_k`, `chat_kb_similarity_threshold`.

### Sync Pipeline (`rag/sync.py`)

```
list_drive_files() → GET api-gateway/storage/files  (all subfolders, category per file)
    → diff against kb_sources (skip unchanged files, detect deletions)
    → download_file() → GET api-gateway/storage/files/{id}/content
    → parse_content() (PDF/DOCX/text)
    → chunk_text()
    → embed_documents() in batches of 96
    → atomic transaction: DELETE old chunks + INSERT new chunks (with source_category)
    → upsert kb_sources (last_synced, chunk_count, status)
```

`sync_drive(force=False)` — smart incremental by default. `force=True` re-syncs everything.
Files needing sync are processed concurrently, bounded by `kb_sync_concurrency` (default 4).
KB subfolders: `general`, `projects`, `purdue`, `career`, `reference`.

### LLM Calls (`llm/gateway.py`)

All generation goes through `AIGateway.chat()` → api-gateway `/ai/v1/chat/completions`. The knowledge-base never calls Anthropic directly. `AIGateway` is a synchronous client (httpx); it is initialised once in `app/main.py` and accessed via a module-level global.

## Key Conventions

- **Ruff** for linting: line length 120, target Python 3.11
- **black** for formatting: line length 120
- All DB access via `asyncpg` pool from `core/database.py`; get with `get_pool()`
- All Drive access via httpx to the api-gateway (never directly to Google APIs)
- Voyage AI client is synchronous; wrapped in `asyncio.to_thread` in `embedder.py` and `reranker.py`
- Config singleton: `from core.config import get_config; config = get_config()`
