# Project Mnemosyne: Implementation Checklist

- [x] **Phase 0: Architecture & Planning**
  - [x] Define dual-tier RAG (Library vs Journal)
  - [x] Finalize model selection (Nomic + BGE)
  - [x] Define stateless API strategy (`use_context=False`)
  - [x] Address security & safety (Delete checks, Auth notes)

- [ ] **Phase 1: Foundation (Config & Registry)**
  - [x] **1A. Model Registry**
    - [x] Create `core/model_registry.py` with `ModelMetadata` schema
    - [x] Populate `MODELS` dict with Nomic, BGE, and Llama definitions
    - [x] Implement `get_models_by_tag` helper
  - [x] **1B. Configuration Updates**
    - [x] Add `hardware_mode` ("gpu"/"cpu") to `core/config.py`
    - [x] Add `model_library` and `model_journal` fields (defaulting to Registry values)
    - [x] Rename `blob_storage_path` default to `./data/library_blob`
  - [x] **1C. Verification**
    - [x] Check `GET /v1/config` returns new hardware/model settings

- [ ] **Phase 2: The Journal Engine (Storage)**
  - [x] **2A. Core Class**
    - [x] Create `rag/journal.py` with `JournalManager` stub
    - [x] Initialize `VectorStore` client within Manager
  - [x] **2B. Storage Logic**
    - [x] Implement `add_entry` (Async: Embed -> Struct -> Upsert)
    - [x] Implement `generate_session_name` (LLM call logic)
  - [x] **2C. Retrieval Logic**
    - [x] Implement `get_recent_context` (Search by session_id)
    - [x] Implement `delete_session` (Delete by payload filter)

- [ ] **Phase 3: Context Pipeline (Execution)**
  - [x] **3A. Engine Refactor**
    - [x] Rename `BasicRAG` to `ContextEngine`
    - [x] Configure `ContextEngine` to hold both Library and Journal retrievers
  - [x] **3B. Service Layer**
    - [x] Add `use_context` arg to `ChatService`
    - [x] Implement "Hybrid Retrieval" (Profile + Library + Journal)
  - [x] **3C. Wiring**
    - [x] Update `app/routes/llm.py` to pass `use_context` flag

- [ ] **Phase 4: API & Data Management**
  - [x] **4A. Memory Endpoints**
    - [x] Create `DELETE /v1/memory` with `?confirm=true` check
    - [x] Create `GET /v1/memory/sessions` listing
  - [x] **4B. Profile & Ingest**
    - [x] Create `GET/PATCH /v1/profile`
    - [x] Create `POST /v1/ingest/file/{filename}` (Manual blob ingest)

- [ ] **Phase 5: Verification & Ingest Library**
  - [ ] **5A. Ingest Library**
    - [ ] Write `scripts/ingest_library.py` (Iterate library_blob -> Ingest)
    - [ ] Run ingest for Nomic model
  - [ ] **5B. Testing**
    - [ ] Verify Chat History recall
    - [ ] Verify Stateless Mode (`use_context=False`)
