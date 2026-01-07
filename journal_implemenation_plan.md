# Journal Implementation Plan

## Overview

Refactor the journal system to mirror the library architecture:
- **Save messages in real-time** to SQLite (not Qdrant)
- **Ingest to RAG on trigger** (manual or auto on session switch)
- **Export sessions** to `journal_blob/` as JSON files before ingestion

This enables chat history to work like ChatGPT (select past sessions) while maintaining RAG functionality for cross-conversation context retrieval.

---

## Design Decisions

| Decision | Choice |
|----------|--------|
| Message Storage | Hybrid: SQLite for active storage, export to `journal_blob/` for RAG |
| Ingestion Trigger | Manual + Auto on session switch (if new content) |
| Re-ingestion Strategy | Delete + Re-ingest entire session |
| Session Title | Truncated first user message (25 chars), set after first assistant response |
| Session Switch Detection | Use `session_id` in chat request - different ID triggers ingest for previous |
| Unfinished Sessions | Accept it - manual ingest later if app closed mid-conversation |

---

## Data Flow

```
Chat Message
     |
     v
SQLite (messages table) -----> Session updated (last_activity, message_count)
     |                                    |
     |                         [On first assistant response]
     |                                    |
     |                                    v
     |                         Auto-set session name (truncated first message)
     |
     |-----> [Trigger: manual OR session switch with new content]
                    |
                    v
          Check: has messages AND (never_ingested OR has_new_messages)
                    |
                    v
          Export session -> journal_blob/{session_id}.json
                    |
                    v
          Delete old chunks from Qdrant (by session_id)
                    |
                    v
          Chunk -> Embed -> Store in Qdrant (journal_entries collection)
                    |
                    v
          Update session (ingested_at = now)
```

---

## Database Schema Changes

### sessions table (MODIFY)

Add `ingested_at` column:

```sql
ALTER TABLE sessions ADD COLUMN ingested_at TEXT;
```

Full schema:
```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    name TEXT,                    -- Auto-set: truncated first user message (25 chars)
    created_at TEXT NOT NULL,
    last_activity TEXT NOT NULL,
    message_count INTEGER DEFAULT 0,
    ingested_at TEXT              -- NEW: When last ingested to RAG (NULL = never)
);
```

### messages table (NEW)

```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,           -- 'user' or 'assistant'
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX idx_messages_session ON messages(session_id);
CREATE INDEX idx_messages_timestamp ON messages(session_id, timestamp);
```

---

## File Storage

### journal_blob/ Directory

Location: `./data/journal_blob/`

Each exported session stored as: `{session_id}.json`

```json
{
  "session_id": "abc-123-def-456",
  "name": "Python async questions",
  "created_at": "2024-01-15T10:30:00Z",
  "exported_at": "2024-01-15T12:00:00Z",
  "messages": [
    {
      "role": "user",
      "content": "How do I use async/await in Python?",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    {
      "role": "assistant",
      "content": "To use async/await in Python, you first need to...",
      "timestamp": "2024-01-15T10:30:05Z"
    }
  ]
}
```

---

## API Endpoints

### New Endpoints

#### GET /v1/memory/sessions/{session_id}/messages

Get a session with all its messages (for loading past conversations).

**Response:**
```json
{
  "session_id": "abc-123",
  "name": "Python async questions",
  "created_at": "2024-01-15T10:30:00Z",
  "last_activity": "2024-01-15T11:45:00Z",
  "message_count": 12,

  "ingestion_status": {
    "ingested": true,
    "ingested_at": "2024-01-15T11:50:00Z",
    "has_new_messages": false,
    "chunk_count": 8
  },

  "messages": [
    {
      "role": "user",
      "content": "How do I use async/await in Python?",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    {
      "role": "assistant",
      "content": "To use async/await in Python...",
      "timestamp": "2024-01-15T10:30:05Z"
    }
  ],

  "request_id": "req_xxx"
}
```

#### POST /v1/memory/sessions/{session_id}/ingest

Manually trigger ingestion of a session into the journal RAG collection.

**Response:**
```json
{
  "session_id": "abc-123",
  "status": "ingested",
  "chunks_created": 8,
  "blob_path": "data/journal_blob/abc-123.json",
  "ingested_at": "2024-01-15T12:00:00Z",
  "request_id": "req_xxx"
}
```

### Modified Endpoints

#### POST /v1/chat/completions

Add logic to:
1. Save user message to SQLite messages table
2. Save assistant response to SQLite messages table
3. Detect session switch (track last session_id)
4. Auto-ingest previous session if switched AND has new content
5. Auto-set session name after first assistant response (if name is NULL)

---

## Code Changes

### File: core/session_store.py

**Modifications:**
- Add `ingested_at` column to sessions table schema
- Add `messages` table schema
- Add methods:
  - `add_message(session_id, role, content)` - Save a chat message
  - `get_messages(session_id)` - Get all messages for a session
  - `get_session_with_messages(session_id)` - Get session + messages in one call
  - `set_ingested_at(session_id, timestamp)` - Mark session as ingested
  - `has_new_messages_since_ingest(session_id)` - Check if needs re-ingest
  - `set_session_name(session_id, name)` - Already exists, will use for auto-naming

### File: core/file_storage.py

**Add:**
- `JournalBlobStorage` class (similar to `BlobStorage` but for journal exports)
- Methods:
  - `export_session(session_id, session_data)` - Write session JSON to journal_blob/
  - `get_session_blob(session_id)` - Read session JSON
  - `delete_session_blob(session_id)` - Remove exported file
  - `list_session_blobs()` - List all exported sessions

### File: rag/journal.py

**Refactor:**
- Remove immediate Qdrant storage from `add_entry()`
- Add new methods:
  - `ingest_session(session_id)` - Full ingestion pipeline:
    1. Get messages from SQLite
    2. Export to journal_blob/
    3. Delete existing chunks from Qdrant (by session_id)
    4. Chunk the conversation
    5. Embed and store in Qdrant
    6. Update ingested_at in sessions table
  - `delete_session_chunks(session_id)` - Remove chunks from Qdrant
  - `get_session_chunk_count(session_id)` - Count chunks in Qdrant for session

**Keep:**
- `get_recent_context()` - Still queries Qdrant for RAG retrieval
- `delete_session()` - Modify to also delete blob and SQLite messages

### File: app/routes/memory.py

**Add:**
- `GET /v1/memory/sessions/{session_id}/messages` endpoint
- `POST /v1/memory/sessions/{session_id}/ingest` endpoint

**Modify:**
- `DELETE /v1/memory/sessions/{session_id}` - Also delete messages and blob

### File: app/routes/llm.py (or create app/routes/chat.py)

**Add:**
- Module-level variable to track last session_id: `_last_session_id: Optional[str] = None`
- After successful chat completion:
  1. Save user message to SQLite
  2. Save assistant message to SQLite
  3. Check if session switched (current != last)
  4. If switched AND last session has new content: trigger ingest
  5. Update `_last_session_id`
  6. If session name is NULL and this is first assistant response: set name

### File: core/config.py

**Add:**
```python
journal_blob_storage_path: str = Field(
    default="./data/journal_blob",
    description="Path to store exported journal session files"
)

journal_title_max_length: int = Field(
    default=25,
    description="Maximum length for auto-generated session titles"
)
```

---

## Implementation Order

### Phase 1: Database & Storage Foundation
1. Update `core/session_store.py`:
   - Add `ingested_at` column migration
   - Add `messages` table
   - Add message CRUD methods
2. Add `JournalBlobStorage` to `core/file_storage.py`
3. Update `core/config.py` with new settings

### Phase 2: Journal Refactor
4. Refactor `rag/journal.py`:
   - Remove immediate Qdrant storage
   - Add `ingest_session()` pipeline
   - Add chunk management methods

### Phase 3: API Endpoints
5. Add new endpoints to `app/routes/memory.py`:
   - GET messages endpoint
   - POST ingest endpoint
6. Modify DELETE session to clean up all data

### Phase 4: Chat Integration
7. Modify `app/routes/llm.py`:
   - Add message saving after chat completion
   - Add session switch detection
   - Add auto-ingest trigger
   - Add auto-naming logic

### Phase 5: Testing & Cleanup
8. Test full flow:
   - New session -> chat -> switch session -> verify ingest
   - Load past session -> verify messages display
   - Manual ingest -> verify RAG retrieval
9. Remove deprecated code from old journal implementation
10. Update any CLI commands if needed

---

## Auto-Ingest Trigger Logic

```python
# In chat completions endpoint, after saving messages:

def maybe_auto_ingest(current_session_id: str):
    global _last_session_id

    if _last_session_id is None:
        # First request, just track
        _last_session_id = current_session_id
        return

    if current_session_id == _last_session_id:
        # Same session, no action
        return

    # Session switched - check if previous session needs ingest
    session_store = get_session_store()

    if session_store.has_new_messages_since_ingest(_last_session_id):
        # Trigger ingest for previous session
        journal_manager = get_journal_manager()
        journal_manager.ingest_session(_last_session_id)

    # Update tracking
    _last_session_id = current_session_id
```

---

## Session Name Auto-Generation

```python
# In chat completions endpoint, after saving assistant response:

def maybe_set_session_name(session_id: str, first_user_message: str):
    session_store = get_session_store()
    session = session_store.get_session(session_id)

    if session and session.get("name") is None:
        # First assistant response - set name from first user message
        truncated = first_user_message[:25]
        if len(first_user_message) > 25:
            truncated = truncated.rstrip() + "..."
        session_store.set_session_name(session_id, truncated)
```

---

## Qdrant Payload Structure (Journal)

When ingesting to Qdrant, each chunk will have:

```python
PointStruct(
    id=str(uuid.uuid4()),
    vector=embedding,
    payload={
        "text": chunk_content,           # The chunked text
        "session_id": "abc-123",         # Links back to session
        "session_name": "Python async...", # For display in retrieval results
        "chunk_index": 0,                # Position in conversation
        "message_count": 12,             # Total messages in session at ingest time
        "ingested_at": "2024-01-15..."   # When this was ingested
    }
)
```

---

## Migration Notes

- Existing Qdrant journal_entries collection may have old format data
- Consider clearing collection on first run of new system, OR
- Handle both old (per-message) and new (per-session chunks) formats in retrieval

---

## Future Considerations (Out of Scope for Now)

- [ ] Startup check: Auto-ingest sessions that were never ingested
- [ ] Background worker for journal ingestion (like library)
- [ ] Incremental ingestion (only new messages) instead of full re-ingest
- [ ] Session export to markdown/text formats
- [ ] Session search by name/content (before loading)
- [ ] Session archiving/soft delete
