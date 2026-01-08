# RAG System Improvements

Incremental improvements to the RAG pipeline focusing on retrieval quality, flexible prompting, and query handling.

## User Review Required

> [!IMPORTANT]
> **Breaking Change**: Enabling BGE-M3 sparse vectors will change the Qdrant collection schema. Clear existing collections before proceeding.

---

## Proposed Changes

### Phase 0: Code Consolidation (Cleanup) ✅ COMPLETE

Address existing code duplication before adding new features.

#### Redundancy 1: Duplicate Chunking

Both `document_ingester.py` and `journal.py` have their own `_chunk_text()` methods with similar logic.

**Solution:** Create shared chunking module.

#### [NEW] [chunking.py](file:///c:/Users/aatha/Desktop/Local%20Git%20Repositories/MY-AI/MY-AI/rag/chunking.py)

```python
def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 100,
    prefer_paragraph_breaks: bool = True
) -> list[str]:
    """Shared chunking logic for Library and Journal."""
    ...
```

#### [MODIFY] document_ingester.py and journal.py

- Remove duplicate `_chunk_text()` methods
- Import from `rag.chunking`

---

#### Redundancy 2: Duplicate Embedding Model Loading

Both `DocumentRetriever` and `JournalManager` load SentenceTransformer separately:
- `retriever.py` line 20-21: loads model
- `journal.py` line 71-72: loads same model again

**Problem:** Two instances of the same large model in memory.

**Solution:** Share the embedder instance.

#### [MODIFY] [rag_setup.py](file:///c:/Users/aatha/Desktop/Local%20Git%20Repositories/MY-AI/MY-AI/rag/rag_setup.py)

Pass the `DocumentRetriever` instance to `JournalManager`:
```python
self._journal = JournalManager(
    vector_store=self.vector_store,
    embedder=self.retriever  # Share the embedder
)
```

#### [MODIFY] [journal.py](file:///c:/Users/aatha/Desktop/Local%20Git%20Repositories/MY-AI/MY-AI/rag/journal.py)

- Accept optional `embedder` parameter in `__init__`
- Use shared embedder if provided, lazy-load only as fallback

---

### Phase 1: Hybrid Search with BGE-M3 ✅ COMPLETE

Enable sparse embeddings from BGE-M3 for hybrid (semantic + lexical) retrieval.

#### [MODIFY] [retriever.py](file:///c:/Users/aatha/Desktop/Local%20Git%20Repositories/MY-AI/MY-AI/rag/retriever.py)

- Add `FlagEmbedding` import for native BGE-M3 hybrid support
- Extend `encode_documents()` to return both dense and sparse vectors
- Add `encode_query()` sparse variant
- Store sparse weights dict alongside dense embeddings

#### [MODIFY] [vector_store.py](file:///c:/Users/aatha/Desktop/Local%20Git%20Repositories/MY-AI/MY-AI/rag/vector_store.py)

- Update `setup_collection()` to configure sparse vector index
- Modify `add_points()` to store both dense and sparse vectors
- Implement `hybrid_search()` using Qdrant's fusion (RRF or weighted)

#### [MODIFY] [rag_setup.py](file:///c:/Users/aatha/Desktop/Local%20Git%20Repositories/MY-AI/MY-AI/rag/rag_setup.py)

- Update `search()` to use hybrid search
- Add config toggle: `use_hybrid_search: bool`

---

### Phase 2: Cross-Encoder Reranking ✅ COMPLETE

Two-stage retrieval: fast vector search → accurate reranking.

#### [NEW] [reranker.py](file:///c:/Users/aatha/Desktop/Local%20Git%20Repositories/MY-AI/MY-AI/rag/reranker.py)

```python
class CrossEncoderReranker:
    """Reranks search results using a cross-encoder model."""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        ...
    
    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[tuple[str, float]]:
        """Rerank documents by relevance to query."""
        ...
```

#### [MODIFY] [rag_setup.py](file:///c:/Users/aatha/Desktop/Local%20Git%20Repositories/MY-AI/MY-AI/rag/rag_setup.py)

- Integrate reranker into `search()` pipeline
- Config: `rerank_enabled: bool`, `rerank_top_k: int`

#### [MODIFY] [config.py](file:///c:/Users/aatha/Desktop/Local%20Git%20Repositories/MY-AI/MY-AI/core/config.py)

Add reranking configuration:
```python
rerank_enabled: bool = Field(default=True)
rerank_candidates: int = Field(default=30)  # Fetch this many, rerank to top_k
rerank_model: str = Field(default="BAAI/bge-reranker-v2-m3")
```

---

### Phase 3: Semantic Chunking ✅ COMPLETE

Replace character-based chunking with semantic chunking in both Library and Journal.

#### [MODIFY] [document_ingester.py](file:///c:/Users/aatha/Desktop/Local%20Git%20Repositories/MY-AI/MY-AI/rag/document_ingester.py)

- Replace `_chunk_text()` with LangChain's `RecursiveCharacterTextSplitter`
- For `.md` files: additionally use `MarkdownHeaderTextSplitter` to extract `section_title`
- For other files (txt, pdf, docx): use base splitter with paragraph/sentence separators

#### [MODIFY] [journal.py](file:///c:/Users/aatha/Desktop/Local%20Git%20Repositories/MY-AI/MY-AI/rag/journal.py)

- Update `_chunk_text()` to use same semantic chunking approach
- Import shared chunking utility from document_ingester (or new `chunking.py` module)

---

### Phase 4: Query Expansion ✅ COMPLETE

LLM-based query rewriting for vague queries.

#### [NEW] [query_processor.py](file:///c:/Users/aatha/Desktop/Local%20Git%20Repositories/MY-AI/MY-AI/rag/query_processor.py)

```python
class QueryProcessor:
    """Processes and expands queries before retrieval."""
    
    def __init__(self, model: str = None):
        # Uses config.query_expansion_model if not specified
        ...
    
    async def process(self, query: str, context: str = None) -> str:
        """Rewrite vague queries using configured LLM."""
        ...
```

#### [MODIFY] [config.py](file:///c:/Users/aatha/Desktop/Local%20Git%20Repositories/MY-AI/MY-AI/core/config.py)

```python
# Query expansion settings
query_expansion_enabled: bool = Field(default=True)
query_expansion_model: str = Field(default="llama3.2:1b")  # Fast model for rewriting
```

#### [MODIFY] [rag_setup.py](file:///c:/Users/aatha/Desktop/Local%20Git%20Repositories/MY-AI/MY-AI/rag/rag_setup.py)

- Integrate query processor before search
- Config: `query_expansion_enabled: bool`, `query_expansion_model: str`

---

### Phase 5: Dynamic System Prompt ✅ COMPLETE

Make system prompt editable via API. Uses existing file-based prompt storage.

#### [NEW] [prompt_manager.py](file:///c:/Users/aatha/Desktop/Local%20Git%20Repositories/MY-AI/MY-AI/core/prompt_manager.py)

```python
class PromptManager:
    """Manages system prompt with override capability."""
    
    def get_system_prompt(self) -> str:
        """Get active system prompt (custom override or default from prompts/llm.md)."""
        # Check for user override in data/prompts/custom_system.md
        # Fall back to core/prompts/llm.md
        ...
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set custom system prompt (persisted to data/prompts/custom_system.md)."""
        ...
```

#### [MODIFY] [profile_manager.py](file:///c:/Users/aatha/Desktop/Local%20Git%20Repositories/MY-AI/MY-AI/core/profile_manager.py)

Existing `ProfileManager` already handles profile persistence. Add profile context formatting:
```python
def get_context_string(self) -> Optional[str]:
    """Format profile as context string for injection, or None if empty."""
    profile = self.get_profile()
    if profile.get("bio") or profile.get("preferences"):
        return f"User: {profile['name']}\nBio: {profile.get('bio', '')}\n..."
    return None
```

#### [NEW ENDPOINT] in [config.py](file:///c:/Users/aatha/Desktop/Local%20Git%20Repositories/MY-AI/MY-AI/app/routes/config.py)

```
GET  /v1/config/prompts/system     → Get current system prompt
PUT  /v1/config/prompts/system     → Update system prompt
```

#### [MODIFY] [llm.py](file:///c:/Users/aatha/Desktop/Local%20Git%20Repositories/MY-AI/MY-AI/app/routes/llm.py)

- Use `PromptManager` for system prompt (line 311 currently hardcoded)
- Inject profile context via `ProfileManager.get_context_string()` if not None

---

### Phase 6: Metadata Enrichment ✅ COMPLETE

Add `document_type`, `ingested_at`, `tags`, and `section_title` to payloads.

#### [MODIFY] [retriever.py](file:///c:/Users/aatha/Desktop/Local%20Git%20Repositories/MY-AI/MY-AI/rag/retriever.py)

Extend `create_points()` metadata:
```python
payload = {
    "text": doc,
    "document_type": metadata.get("document_type"),
    "ingested_at": datetime.utcnow().isoformat(),
    "tags": metadata.get("tags", []),
    "section_title": metadata.get("section_title"),
    # ... existing fields
}
```

#### [MODIFY] [workers.py](file:///c:/Users/aatha/Desktop/Local%20Git%20Repositories/MY-AI/MY-AI/rag/workers.py)

Pass `document_type` from parsed file extension.

#### [MODIFY] Upload API

Allow optional `tags` parameter on file upload.

---

## Verification

Manual testing after each phase:
```bash
make dev
curl -X POST localhost:8000/v1/ingest/upload -F "file=@test.md"
curl -X POST localhost:8000/v1/chat/completions -d '{"messages": [{"role": "user", "content": "test query"}]}'
```

---

## Implementation Order

| Phase | Depends On | Effort | Priority |
|-------|------------|--------|----------|
| 0. Code Consolidation | - | Low | Highest |
| 1. Hybrid Search | Phase 0 | Medium | High |
| 2. Cross-Encoder | Phase 1 | Low | High |
| 3. Semantic Chunking | Phase 0 | Low | Medium |
| 4. Query Expansion | - | Medium | Medium |
| 5. Dynamic Prompts | - | Low | Medium |
| 6. Metadata | - | Low | Low |

**Recommended start**: Phase 0 (consolidation) then Phase 1 (Hybrid Search).
