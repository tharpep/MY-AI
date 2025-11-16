# Step 0 Deliverable — Contracts & Guardrails (Template)

This is the **planning artifact** for the Personal AI project. It defines the contracts, guardrails, and intended behavior before any code is written. It should be one page, conceptual only.

---

## 1. Surfaces (APIs)

**Personal AI Assistant API (Unified)**

- **Smart Chatbot Endpoint:**
  - `POST /v1/chat/completions` - **Intelligent chatbot** that automatically uses RAG when appropriate
  - **Intent analysis** - Determines when to use document context vs general knowledge
  - **Seamless experience** - User doesn't need to choose between RAG and pure AI
- **OpenAI-Compatible Endpoints:**
  - `POST /v1/embeddings` - Generate embeddings (OpenAI format)
  - `GET /v1/models` - List available models
- **RAG & Document Management:**
  - `POST /v1/query` - Direct RAG-powered question answering with citations
  - `POST /v1/ingest` - Ingest documents into knowledge base for RAG
  - `GET /v1/stats` - RAG system statistics
- **Health & Monitoring:**
  - `GET /health/` - Basic health check
  - `GET /health/detailed` - Detailed health with component status

**Architecture Notes:**
- **Single unified API** - No separate LLM Gateway service
- **Smart routing** - Chatbot automatically decides when to use RAG
- **Local-first design** - Everything runs in one process
- **OpenAI compatibility** - Drop-in replacement for OpenAI API
- **RAG integration** - Documents ingested become searchable knowledge base
- **Streaming: off for v0** (SSE later)
- **Models: local-first** (Ollama), **external fallback** (Purdue GenAI Studio)
- **Auth: local dev none**; **external connectors** require bearer tokens
- **Response envelope:** minimal (`answer` + `citations`), optional `debug=true` for transcripts

---

## 2. Smart Chatbot Routing (Tier‑2)

- **Intent Analysis** - Analyze user messages to determine if RAG context is needed
- **Automatic RAG Integration** - Chatbot seamlessly uses document context when appropriate
- **Smart Routing Logic:**
  - **General questions** ("What's the weather?") → Pure AI response
  - **Knowledge questions** ("What did I work on?") → RAG + AI response with citations
  - **Uncertainty** → Ask one clarifying question, then fallback to RAG
- **Internal routing** - No separate LLM Gateway service calls
- **Direct integration** - RAG system calls LLM Gateway class methods directly
- **Tool routing** - Must output strict JSON: `{ tool: str, args: {...} }`
- **Allowlist controls** which tools are eligible
- **Default tool:** `rag_answer` when intent is knowledge lookup or when uncertainty remains
- **No loops.** Optional final LLM call to **summarize/phrase** output. Default = **off** (return raw RAG/tool results). Toggleable with per‑request flag `summarize=true|false`.

---

## 3. Hybrid Memory System (RAG + Persistent Memory + Context)

**Three-Layer Knowledge Architecture:**

**1. RAG (Document-Based Knowledge):**
- **Purpose**: Search through ingested documents when relevant
- **Content**: Your notes, emails, documents, project files, research papers
- **Benefits**: Always up-to-date, can cite sources, handles large information
- **Use Cases**: "What did I work on yesterday?", "Summarize my meeting notes"

**2. Persistent Memory (Personal Facts & History):**
- **Purpose**: Store information about you across sessions
- **Content**: Personal facts, preferences, conversation history, ongoing projects
- **Benefits**: Remembers you, builds relationship, maintains context
- **Use Cases**: "Continue our discussion about RAG", "What are my current projects?"

**3. Context Awareness (Current Session):**
- **Purpose**: Maintain conversation flow and immediate context
- **Content**: Current conversation state, recent messages, active topics
- **Benefits**: Natural conversation flow, immediate relevance
- **Use Cases**: "What were we just discussing?", "Build on that idea"

**Smart Integration:**
- **Intent analysis** determines which knowledge layer(s) to use
- **Seamless experience** - user doesn't need to specify knowledge source
- **Hybrid responses** - combines multiple knowledge sources when relevant

---

## 4. Local CLI Interaction (No API Required)

**CLI - Direct Local Interaction:**
- **`poetry run my-ai demo rag`** - Run RAG demo with automated queries
- **`poetry run my-ai demo llm`** - Run LLM demo with automated questions
- **`poetry run my-ai demo tuning`** - Manage and test fine-tuned models
- **`poetry run my-ai demo api`** - Start the FastAPI server for web/frontend access

**Benefits of CLI Mode:**
- **No server overhead** - Direct Python execution
- **Faster iteration** - Quick testing and debugging
- **Local privacy** - Everything stays on your machine
- **Development friendly** - Perfect for testing features

**Use Cases:**
- **Development** - Test RAG queries, tune models, debug issues
- **Quick questions** - Ask about your documents without starting a server
- **Model testing** - Try different models and configurations
- **Documentation** - Learn how the system works

---

## 5. Deployment Strategy (Local + Your Own Cloud)

**Local Development:**
- **Hardware-aware** - Automatically uses appropriate models (laptop vs dedicated GPU)
- **Fast iteration** - Quick testing and development
- **Privacy** - Everything stays on your machine
- **CLI access** - Direct interaction via `poetry run my-ai demo`

**Your Own Cloud Production:**
- **Deploy anywhere** - AWS, GCP, Azure, DigitalOcean, VPS, etc.
- **Access anywhere** - Phone, laptop, work computer
- **More power** - Better hardware than your laptop
- **Always available** - 24/7 access to your AI assistant
- **Still private** - Your data, your control, your cloud

**Deployment Options:**
- **Docker** - `docker build` and deploy to any cloud
- **Cloud platforms** - Railway, Render, Fly.io for easy deployment
- **VPS** - Your own server for maximum control
- **Hybrid** - Local development, cloud production

---

## 6. Planned Tool Allowlist (initial)

**Phase v0 (launch, read‑only):**
- `rag_answer` — default knowledge lookup with citations
- `drive_search` — search files by query/owner/mime; returns URIs + snippets (respect ACL)
- `web_search` — fetch public references for context only; returns links/snippets (with attribution)

**Phase v1 (first connectors, read‑only):**
- `calendar_lookup` — read events by date/range; returns normalized list of events

**Phase v2 (deferred; scope & safety review required; read‑only by default):**
- `spotify_lookup` — read playlists/tracks (user library allowed via scopes)
- `banking_readonly` — Plaid integration for balances/transactions (feature‑flagged OFF by default; strong auth & redaction policies)

*(Tools listed here are intentions only — no implementations yet.)***

---

## 4. Tool Contract Templates (fill one per tool)

### A) `rag_answer`
- **Purpose:** Retrieve and cite answers from the ingested document corpus (RAG knowledge base).
- **Inputs (typed):** `query: str` · `top_k: int (default 5)` · `filters: dict (optional)`
- **Output (typed):** `answer_snippets: [ {chunk_id, doc_uri, text, score} ]` · `citations: [doc_uri]`
- **Side effects:** none (read‑only)
- **Idempotent:** yes
- **Safety notes:** always return citations; no model synthesis when `summarize=false`.
- **Document ingestion:** Documents ingested via `/v1/ingest` become part of the searchable RAG corpus.

### B) `drive_search` (v1)
- **Purpose:** Find files in connected drive(s).
- **Inputs:** `query: str` · `owner: str (optional)` · `mime: str (optional)` · `limit: int (default 10)`
- **Output:** `files: [ {uri, title, snippet, owner, modified_at} ]`
- **Side effects:** external API call (read‑only)
- **Idempotent:** yes
- **Safety notes:** respect ACL; redact PII in logs.

### C) `calendar_lookup` (v1)
- **Purpose:** Read user calendar for a date/range.
- **Inputs:** `email: str` · `date: ISO8601 | {start,end}` · `time_zone: str (optional)`
- **Output:** `events: [ {title, start, end, location, attendees[]} ]`
- **Side effects:** external API call (read‑only)
- **Idempotent:** yes
- **Safety notes:** PII handling; do not store raw invite bodies.

### D) `web_search` (v1)
- **Purpose:** Retrieve public web results with attribution.
- **Inputs:** `query: str` · `recency_days: int (optional)` · `limit: int (default 5)`
- **Output:** `results: [ {title, url, snippet, published_at?} ]`
- **Side effects:** external call (read‑only)
- **Idempotent:** yes
- **Safety notes:** mark sources clearly; avoid paywalled links by default.

### E) `spotify_lookup` (v2 — deferred)
- **Purpose:** Read Spotify data (playlists, tracks, artists) to support organization or recommendations.
- **Inputs:** `entity: enum['playlist','track','artist']` · `query: str` · `limit: int (default 10)` · `user_scope: bool (default true for user library)`
- **Output:**
  - `playlists: [ {id, name, owner, track_count, uri} ]` *or*
  - `tracks: [ {id, name, artist, album, uri, duration_ms} ]` *or*
  - `artists: [ {id, name, uri, genres[]} ]`
- **Side effects:** external API call (read‑only)
- **Idempotent:** yes
- **Auth scope:** app‑only search; **user‑library scopes** when `user_scope=true`
- **Safety notes:** never store refresh tokens in logs; redact usernames/emails; rate‑limit requests

### F) `banking_readonly` (v2 — deferred; sensitive; **OFF by default**)
- **Purpose:** Read balances/transactions from a linked financial account **read‑only** for budgeting insights (no transfers).
- **Inputs:** `account_selector: enum['all','by_id']` · `account_id: str (required if 'by_id')` · `range: {start, end}`
- **Output:** `accounts: [ {id, name, type, current_balance, available_balance, currency} ]` · `transactions: [ {id, account_id, date, description, amount, category} ]`
- **Side effects:** external API call (read‑only)
- **Idempotent:** yes (results may vary as balances update)
- **Auth scope:** **Plaid** provider token with **read** scopes only
- **Safety notes:**
  - **PII/financial redaction:** mask account numbers; truncate descriptions; never log raw tokens
  - **Storage:** do not persist full transaction bodies by default; cache minimal aggregates with TTL
  - **Compliance:** require explicit user consent and per‑user scoping; **feature‑flagged OFF by default**

---

## 5. Validation & Error Semantics

**Router plan validation**
- Router must return strict JSON `{ tool: string, args: object }`.
- Enforce JSON Schema for the chosen tool’s inputs (types, required fields, ranges).
- Unknown tool (not on allowlist) → reject.
- On validation failure → **422 Unprocessable** with a `clarify` object listing `missing_fields`, `bad_types`, `out_of_range`.

**Clarifying flow**
- Ask **one** clarifying question when required fields are missing/ambiguous.
- If still uncertain after one round, **fallback to `rag_answer`** (read‑only) rather than guessing.

**Tool input/output validation**
- Validate tool outputs against their output schemas before composing a response.
- Redact secrets/PII (access tokens, account numbers, emails) from outputs, transcripts, and logs per the PII policy.

**Auth & scopes**
- Personal API (local dev): no auth.
- External connectors/LLMs: require **bearer tokens** with appropriate scopes; missing/invalid → **401**; insufficient permissions → **403**.

**Idempotency & side‑effects**
- v0 tools are **read‑only** and **idempotent**; safe to retry.
- Future non-idempotent tools must declare side-effects; require explicit user confirmation; **never** auto-retry them.

**Rate limiting**
- Conceptual defaults: 60 req/min/user per tool; `web_search` 10 req/min/IP.
- Exceeding limits → **429** with `retry_after_ms`.

**Retries & backoff**
- Eligible for auto-retry: **429**, **502/503**, and transient **5xx** from upstreams. Use exponential backoff (e.g., 250ms → 1s → 2s; max 3 attempts).
- Do **not** auto-retry on **401/403** or validation errors (**4xx** besides 429).

**Timeout behavior**
- Timeouts (conceptual defaults): chat 15s; embeddings 30s; tool exec 10–20s.
- On timeout → cancel subtask and return a user-safe error with optional `retry_after_ms`.

**Error mapping (summary)**
- **400** malformed request · **401** unauthenticated · **403** forbidden · **404** not found · **409** conflict · **422** invalid plan/args · **429** rate-limited · **500** internal · **502/503** upstream failure.

**Transparency**
- Responses include a `request_id`. Full transcripts/metrics are logged server-side; include inline only when `debug=true`.

---

## 6. Budgets & Limits

**Per-request Gateway call caps**
- **RAG path:** ≤ 2 (embeddings + optional summarization)
- **Tool path:** ≤ 2 (routing + optional summarization)
- **Agent-lite path:** ≤ 3 (routing + 1 step + optional summarization)

**Token caps**
- Input tokens: ≤ *TBD* (set once real model chosen)
- Output tokens: ≤ *TBD*

**Timeouts (service-level)**
- LLM chat: 15s
- Embeddings: 30s
- Tool exec: 10–20s depending on connector
- Overall query SLA target: conceptual only; benchmark before setting targets

**Storage budgets**
- Corpus size target: *TBD* (determine once store selected)
- Retrieval default top-k: 5 (configurable)
- **Memory:** start with **short-term memory** (session window). Later add **long-term condensed memory** (summaries/embeddings). Local machine may store full transcripts; Personal API will persist only condensed/needed state.

**Concurrency & load**
- Default: 10 concurrent queries per user in dev; increase with infra scaling

**Failure budgets**
- Error budget aligned to SLOs (e.g., 99.5% availability monthly)
- Retry budget: max 3 per request for retryable errors

---

## 7. Telemetry & Audit

- **Metrics:** latency p50/p95, gateway_calls, tokens_in/out, retrieval_hit_rate, tool call durations.
- **Transcript spec:** ordered events `[llm_route, retrieval, tool_exec, compose]` with inputs/outputs hashes and durations.
- **Return policy:** default response = `answer|data` + `citations`. Use `debug=true` to include transcript, usage, analytics inline.
- **Logs:** full transcripts and metrics stored server-side with Request ID.
- **PII policy:** redact emails, account IDs, access tokens; store salted hashes where feasible.

---

## 8. Data & Security

- **Provenance per doc:** `source_uri`, `acquired_at`, `version`, `checksum`, `acl_tag`, `owner`.
- **Privacy posture:** local-only by default; cloud fallbacks (external AI APIs, connectors) are opt-in.
- **Secrets handling:** env vars or vault; never log secrets.
- **AuthZ:** default-deny tool allowlist; per-user/project scopes.
- **Memory:**
  - **RAG Memory:** Document-based knowledge stored in vector database (Qdrant)
  - **Persistent Memory:** Personal facts, preferences, conversation history stored locally (SQLite)
  - **Context Memory:** Current session state and conversation flow (in-memory)
  - **Hybrid Integration:** Smart routing between memory layers based on intent analysis
  - User can purge or reset any memory layer at any time.

---

## 9. Non-Functional Requirements

- **Latency targets:** to be benchmarked before setting firm goals.
- **Availability SLO:** target ~99.5% monthly uptime for Personal API once deployed.
- **Extensibility:** adding a tool requires only adding its contract + adapter; no core changes.
- **Portability:** switching local ↔ cloud models is config-only; no code changes.
- **Scalability:** designed to scale to multiple connectors and thousands of docs without architecture change.

---

## 10. Example Flows (conceptual)

- **RAG happy path:** user text → embed → retrieve → extractive compose → (optional) LLM phrase → cited answer.
- **Tool happy path:** user text → LLM plan `{calendar_lookup, args}` → validate → exec → (optional) LLM phrase → answer + transcript.
- **Failure path:** invalid plan → clarify → user supplies field → proceed or fallback to rag_answer.
- **Memory touchpoint:** after each response, session memory updated. Later, condensed summaries written to long-term store.

---

## 11. Open Questions & Risks

- **Open:** streaming support; which tools beyond v0; details of provenance storage; exact memory condensation strategies.
- **Risks:** model hallucinating tools; over-broad permissions; privacy leaks; memory bloat.
- **Mitigations:** strict schema validation, allowlist, human approval for corpus changes, short-term + condensed long-term memory.

---

## 12. Sign-off Checklist

- [ ] Surfaces defined
- [ ] Allowlist agreed
- [ ] Validation & errors documented
- [ ] Budgets set
- [ ] Telemetry & audit fields selected
- [ ] Data provenance & ACLs defined
- [ ] Example flows reviewed
- [ ] Risks accepted / mitigations noted
- [ ] Memory strategy aligned (short-term now, long-term later)

---

