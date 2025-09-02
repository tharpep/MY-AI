# Personal AI — Software Development Path (Tier‑2, Local‑First)

> **Scope:** Software only. Local-first models via LLM Gateway; no hardware here. Keep it conceptual and implementation-agnostic.

---

## 0) Planning — Contracts & Guardrails (no code)
**Goal:** Write down the boundaries before building anything.
- **Surfaces:**
  - **LLM Gateway (AI API):** `chat`, `embeddings` only. No business logic.
  - **Personal API:** `/v1/query` (answers), `/v1/ingest` (documents), internal `/v1/tools/run` (later).
- **Tier‑2 routing stance:** one model call selects `{tool, args}` from an allowlist **or** `rag_answer`.
- **Allowlist (planned, not implemented yet):** e.g., `calendar_lookup`, `drive_search`, `web_search`, `rag_answer`.
- **Schema/validation rules:** Router must return strict JSON matching tool schemas; reject on invalid → ask clarifying Qs.
- **Budgets (conceptual):** RAG ≤ 2 Gateway calls; tools path ≤ 2; no open-ended loops.
- **Telemetry plan:** request IDs, token/latency counters, tool I/O transcript returned to caller.

**Deliverable:** 1‑page doc with endpoint I/O shapes, tool allowlist, validation & error semantics.

---

## 1) Scaffolding — Personal API Skeleton (minimal)
**Goal:** Stand up the product boundary without features.
- Create repo layout; add `/v1/query` and `/healthz` stubs.
- Wire a response wrapper that always returns `answer | data`, `citations`, `transcript` (empty for now).
- Add placeholders for `rag/`, `agents/`, `connectors/` with TODOs only (no tools yet).

**Definition of Done:** API runs; `/v1/query` returns a placeholder with empty transcript.

---

## 2) Local LLM Gateway — Models Only
**Goal:** Provide language intelligence behind a stable contract.
- Stand up local backend (e.g., Ollama/vLLM) via Gateway exposing `chat` & `embeddings`.
- Keep provider‑agnostic: swap to cloud later without changing Personal API code.

**DoD:** Personal API can call Gateway and get a real text/embedding back.

---

## 3) RAG MVP — Read‑Only Corpus
**Goal:** Useful answers from your own docs before tools exist.
- **Ingest:** loaders → chunk → **embed via Gateway** → store (SQLite + FAISS/sqlite‑vec).
- **Query:** embed query → retrieve (BM25 + vectors) → **extractive compose** (citations) → optional LLM summarization.

**DoD:** `/v1/query` answers from seed corpus with citations; zero tools involved.

---

## 4) Tier‑2 Router — AI‑Assisted, Bounded
**Goal:** Flexible intent handling without loops.
- One **Gateway** call per request to return `{tool, args}` from the allowlist **or** `rag_answer`.
- Strict schema validation; invalid → clarify rather than guess.
- Optional final **Gateway** call to phrase results.

**DoD:** Natural language like “What do I have Friday?” routes correctly; total Gateway calls: 1–2.

---

## 5) First Tools/Connectors — Read‑Only
**Goal:** Deterministic tool execution with transparent IO.
- Implement 2–3 tools (e.g., `calendar_lookup`, `drive_search`, `web_search`).
- Normalize outputs into a simple `Result` shape; include raw data + short summary.
- Capture and return **transcripts** (tool name, args, result, timings).

**DoD:** Tool execution works end‑to‑end and is auditable in responses.

---

## 6) Mutable Corpus — AI‑Proposed, Human‑Approved Updates
**Goal:** Let the assistant help maintain knowledge without losing control.
- **Proposal layer:** tools/model produce *diff proposals* (add/update/delete docs/chunks) with provenance.
- **Review:** user approves/rejects proposals; no auto‑commit.
- **Apply:** approved diffs → (re)chunk → (re)embed via Gateway → upsert store; re‑embed only changed chunks.

**DoD:** Approved changes update the store; provenance and version history are visible.

---

## 7) Observability, Evals, Safeguards
**Goal:** Keep quality high as features grow.
- **Metrics:** latency p50/p95, token usage, retrieval precision (top‑k hit rate).
- **Evals:** small golden set (≈15 Q/A) for faithfulness & citation accuracy; run on changes.
- **Budgets & Policies:** cap Gateway calls; tool allowlists by mode/user; deny unknown tools.

**DoD:** Failures surface clearly; transcripts explain what happened and why.

---

## Roles & Boundaries (Always)
- **LLM Gateway:** language intelligence only (chat/embeddings). No connectors.
- **Personal API:** routing, tools, RAG, policies, auth, audit, data lifecycle.
- **You (engineer):** approve corpus changes, define tools/schemas, read transcripts, tune allowlists.

