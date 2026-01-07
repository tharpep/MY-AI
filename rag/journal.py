"""
Journal Manager for Project Mnemosyne
Handles chat history ingestion and retrieval in Qdrant (Tier 2: The Journal).

Architecture:
- Messages are saved to SQLite in real-time (via session_store)
- Sessions are exported to journal_blob/ and ingested to Qdrant on trigger
- RAG retrieval searches the Qdrant journal collection
"""

import logging
import uuid
from datetime import datetime
from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from rag.vector_store import VectorStore
from core.config import get_config
from core.model_registry import get_configured_model

logger = logging.getLogger(__name__)


class JournalEntry(BaseModel):
    """
    Schema for a journal entry stored in Qdrant payload.

    Used for RAG retrieval results.
    """
    role: Literal["user", "assistant"]
    content: str
    session_id: str
    timestamp: str


class JournalChunkPayload(BaseModel):
    """
    Schema for a journal chunk stored in Qdrant.

    Each chunk represents a portion of a conversation session.
    """
    text: str
    session_id: str
    session_name: Optional[str]
    chunk_index: int
    total_chunks: int
    message_count: int
    ingested_at: str


class JournalManager:
    """
    Manages chat history ingestion and retrieval.

    Messages are stored in SQLite (session_store) in real-time.
    Sessions are exported and ingested to Qdrant on trigger for RAG.
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        """
        Initialize the Journal Manager.

        Args:
            vector_store: Optional VectorStore instance. If not provided,
                         creates one using config settings.
        """
        self.config = get_config()

        # Use provided vector store or create new one
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = VectorStore(
                use_persistent=self.config.storage_use_persistent,
                qdrant_host=self.config.qdrant_host,
                qdrant_port=self.config.qdrant_port
            )

        # Get embedding model info from registry
        self.model_info = get_configured_model("journal")
        self.embedding_dim = self.model_info.dimension or 384

        # Embedding model will be initialized lazily
        self._embedder = None

        # Ensure collection exists
        self._setup_collection()

        logger.info(f"JournalManager initialized with model: {self.model_info.name}")

    def _setup_collection(self) -> None:
        """Create journal collection if it doesn't exist."""
        self.vector_store.setup_collection(
            collection_name=self.config.journal_collection_name,
            embedding_dim=self.embedding_dim
        )

    @property
    def embedder(self):
        """Lazy-load the embedding model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.model_info.name)
            logger.info(f"Loaded embedding model: {self.model_info.name}")
        return self._embedder

    # =========================================================================
    # Session Ingestion Methods
    # =========================================================================

    def ingest_session(self, session_id: str) -> Dict[str, Any]:
        """
        Ingest a session into the journal RAG collection.

        Pipeline:
        1. Get session with messages from SQLite
        2. Export to journal_blob/
        3. Delete existing chunks from Qdrant
        4. Chunk the conversation text
        5. Embed and store chunks in Qdrant
        6. Update ingested_at in SQLite

        Args:
            session_id: Session identifier to ingest

        Returns:
            Dict with ingestion results (chunks_created, blob_path, etc.)
        """
        from core.session_store import get_session_store
        from core.file_storage import get_journal_blob_storage

        logger.info(f"Starting ingestion for session: {session_id}")

        # Step 1: Get session with messages
        session_store = get_session_store()
        session_data = session_store.get_session_with_messages(session_id)

        if session_data is None:
            logger.error(f"Session not found: {session_id}")
            return {"error": f"Session not found: {session_id}"}

        messages = session_data.get("messages", [])
        if not messages:
            logger.warning(f"Session has no messages: {session_id}")
            return {"error": "Session has no messages", "session_id": session_id}

        # Step 2: Export to journal_blob/
        blob_storage = get_journal_blob_storage()
        blob_path = blob_storage.export_session(session_id, session_data)
        logger.info(f"Exported session to: {blob_path}")

        # Step 3: Delete existing chunks from Qdrant
        deleted_count = self.delete_session_chunks(session_id)
        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} existing chunks for session {session_id}")

        # Step 4: Format and chunk the conversation
        conversation_text = self._format_conversation_for_ingestion(session_data)
        chunks = self._chunk_text(
            conversation_text,
            chunk_size=self.config.journal_chunk_size,
            overlap=self.config.journal_chunk_overlap
        )
        logger.info(f"Created {len(chunks)} chunks from {len(messages)} messages")

        # Step 5: Embed and store chunks
        ingested_at = datetime.utcnow().isoformat()
        points = []

        for i, chunk_text in enumerate(chunks):
            embedding = self.embedder.encode(chunk_text).tolist()

            payload = JournalChunkPayload(
                text=chunk_text,
                session_id=session_id,
                session_name=session_data.get("name"),
                chunk_index=i,
                total_chunks=len(chunks),
                message_count=len(messages),
                ingested_at=ingested_at
            )

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload.model_dump()
            )
            points.append(point)

        # Batch upsert to Qdrant
        chunks_created = self.vector_store.add_points(
            self.config.journal_collection_name,
            points
        )
        logger.info(f"Stored {chunks_created} chunks in Qdrant")

        # Step 6: Update ingested_at in SQLite
        session_store.set_ingested_at(session_id, ingested_at)

        return {
            "session_id": session_id,
            "chunks_created": chunks_created,
            "blob_path": blob_path,
            "ingested_at": ingested_at,
            "message_count": len(messages)
        }

    def _format_conversation_for_ingestion(self, session_data: Dict[str, Any]) -> str:
        """
        Format a session's messages as text for RAG ingestion.

        Args:
            session_data: Session dict with messages

        Returns:
            Formatted conversation text
        """
        parts = []

        # Add session context header
        if session_data.get("name"):
            parts.append(f"Conversation: {session_data['name']}")
            parts.append("")

        # Format each message
        for msg in session_data.get("messages", []):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            parts.append(f"[{role}] {content}")

        return "\n\n".join(parts)

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 1500,
        overlap: int = 150
    ) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at a paragraph or sentence boundary
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind("\n\n", start, end)
                if para_break > start + chunk_size // 2:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    for sep in [". ", ".\n", "? ", "?\n", "! ", "!\n"]:
                        sent_break = text.rfind(sep, start, end)
                        if sent_break > start + chunk_size // 2:
                            end = sent_break + len(sep)
                            break

            chunks.append(text[start:end].strip())
            start = end - overlap

        return [c for c in chunks if c]  # Filter empty chunks

    # =========================================================================
    # Chunk Management Methods
    # =========================================================================

    def delete_session_chunks(self, session_id: str) -> int:
        """
        Delete all chunks for a session from Qdrant.

        Args:
            session_id: Session identifier

        Returns:
            Number of chunks deleted (approximate)
        """
        try:
            # Count before deletion
            count_before = self.get_session_chunk_count(session_id)

            # Delete by session_id filter
            self.vector_store.client.delete(
                collection_name=self.config.journal_collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="session_id",
                            match=MatchValue(value=session_id)
                        )
                    ]
                )
            )

            logger.info(f"Deleted chunks for session: {session_id}")
            return count_before

        except Exception as e:
            logger.error(f"Failed to delete session chunks: {e}")
            return 0

    def get_session_chunk_count(self, session_id: str) -> int:
        """
        Count chunks in Qdrant for a session.

        Args:
            session_id: Session identifier

        Returns:
            Number of chunks
        """
        try:
            result = self.vector_store.client.count(
                collection_name=self.config.journal_collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="session_id",
                            match=MatchValue(value=session_id)
                        )
                    ]
                )
            )
            return result.count
        except Exception as e:
            logger.error(f"Failed to count session chunks: {e}")
            return 0

    # =========================================================================
    # RAG Retrieval Methods
    # =========================================================================

    def get_context_for_chat(
        self,
        query: str,
        top_k: int,
        similarity_threshold: float,
        session_id: Optional[str] = None
    ) -> List[tuple[str, float]]:
        """
        Get RAG context for chat endpoint (matching Library interface).
        
        Performs vector search with top-k and filters by similarity threshold.
        This is the recommended method for chat context retrieval.
        
        Args:
            query: User query/message
            top_k: Number of entries to retrieve from vector search
            similarity_threshold: Minimum similarity score to include (0.0-1.0)
            session_id: Optional session filter (None = search all sessions)
            
        Returns:
            List of (text, similarity_score) tuples that pass threshold.
            Empty list if no entries pass threshold.
        """
        try:
            # Generate query embedding
            query_vector = self.embedder.encode(query).tolist()
            
            # Build optional filter for session_id
            query_filter = None
            if session_id:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="session_id",
                            match=MatchValue(value=session_id)
                        )
                    ]
                )
            
            # Search Qdrant
            results = self.vector_store.client.query_points(
                collection_name=self.config.journal_collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k
            ).points
            
            # Convert to (text, score) tuples
            retrieved = [(hit.payload.get("text", ""), hit.score) for hit in results]
            
            # Filter by similarity threshold
            filtered = [(text, score) for text, score in retrieved if score >= similarity_threshold]
            
            # Log retrieval details if logging enabled
            if self.config.log_output:
                logger.info(f"Journal Retrieval - Query: '{query[:100]}...'")
                logger.info(f"  Top-K: {top_k}, Threshold: {similarity_threshold}")
                if session_id:
                    logger.info(f"  Session Filter: {session_id}")
                logger.info(f"  Retrieved: {len(retrieved)} entries, Filtered: {len(filtered)} entries")
                if retrieved:
                    logger.info(f"  Retrieved Entries (before threshold filter):")
                    for i, (text, score) in enumerate(retrieved[:5], 1):  # Show top 5
                        text_preview = text[:150] + "..." if len(text) > 150 else text
                        passed = "✓" if score >= similarity_threshold else "✗"
                        logger.info(f"    [{i}] {passed} Score: {score:.3f} | {text_preview}")
                if filtered:
                    logger.info(f"  Entries passing threshold ({len(filtered)}):")
                    for i, (text, score) in enumerate(filtered, 1):
                        text_preview = text[:150] + "..." if len(text) > 150 else text
                        logger.info(f"    [{i}] Score: {score:.3f} | {text_preview}")
                else:
                    if retrieved:
                        max_score = max(score for _, score in retrieved) if retrieved else 0
                        logger.warning(f"  No entries passed threshold (max score: {max_score:.3f} < {similarity_threshold})")
                    else:
                        logger.info(f"  No entries retrieved")
            
            return filtered
            
        except Exception as e:
            logger.error(f"Journal retrieval failed: {e}")
            return []
    
    async def get_recent_context(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> List[JournalEntry]:
        """
        Retrieve relevant chat history for context.

        Searches the ingested journal chunks in Qdrant.

        Args:
            query: Search query for semantic matching
            session_id: Optional session filter (None = search all sessions)
            limit: Maximum entries to return

        Returns:
            List of relevant JournalEntry objects
        """
        try:
            # Generate query embedding
            query_vector = self.embedder.encode(query).tolist()

            # Build optional filter for session_id
            query_filter = None
            if session_id:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="session_id",
                            match=MatchValue(value=session_id)
                        )
                    ]
                )

            # Search Qdrant
            results = self.vector_store.client.query_points(
                collection_name=self.config.journal_collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit
            ).points

            # Convert to JournalEntry objects for compatibility
            entries = []
            for hit in results:
                try:
                    # Handle both old format (per-message) and new format (chunks)
                    payload = hit.payload

                    if "text" in payload:
                        # New chunk format
                        entry = JournalEntry(
                            role="assistant",  # Chunks don't have a single role
                            content=payload.get("text", ""),
                            session_id=payload.get("session_id", ""),
                            timestamp=payload.get("ingested_at", "")
                        )
                    else:
                        # Old per-message format
                        entry = JournalEntry(**payload)

                    entries.append(entry)
                except Exception as e:
                    logger.warning(f"Failed to parse journal entry: {e}")

            return entries

        except Exception as e:
            logger.error(f"Failed to retrieve journal context: {e}")
            return []

    # =========================================================================
    # Session Management Methods
    # =========================================================================

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete all data for a session (Qdrant chunks, blob, SQLite).

        Args:
            session_id: Session identifier to delete

        Returns:
            True if deletion was successful
        """
        from core.session_store import get_session_store
        from core.file_storage import get_journal_blob_storage

        try:
            # Delete chunks from Qdrant
            self.delete_session_chunks(session_id)

            # Delete exported blob
            blob_storage = get_journal_blob_storage()
            blob_storage.delete_session(session_id)

            # Delete from SQLite (session + messages)
            session_store = get_session_store()
            session_store.delete_session(session_id)

            logger.info(f"Deleted all data for session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List sessions from SQLite (authoritative source).

        Args:
            limit: Maximum number of sessions to return

        Returns:
            List of session dicts
        """
        from core.session_store import get_session_store

        try:
            session_store = get_session_store()
            return session_store.list_sessions(limit=limit)
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    async def clear_all(self) -> bool:
        """
        Clear all entries from the journal Qdrant collection.

        Note: This only clears Qdrant, not SQLite or blob storage.

        Returns:
            True if successful
        """
        try:
            self.vector_store.client.delete_collection(self.config.journal_collection_name)
            self._setup_collection()
            logger.info("Cleared all journal entries from Qdrant")
            return True
        except Exception as e:
            logger.error(f"Failed to clear journal: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get journal collection statistics."""
        return self.vector_store.get_collection_stats(self.config.journal_collection_name)

    def get_ingestion_status(self, session_id: str) -> Dict[str, Any]:
        """
        Get ingestion status for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dict with ingestion status info
        """
        from core.session_store import get_session_store
        from core.file_storage import get_journal_blob_storage

        session_store = get_session_store()
        blob_storage = get_journal_blob_storage()

        session = session_store.get_session(session_id)
        if session is None:
            return {"exists": False}

        chunk_count = self.get_session_chunk_count(session_id)
        has_blob = blob_storage.exists(session_id)
        has_new_messages = session_store.has_new_messages_since_ingest(session_id)

        return {
            "exists": True,
            "session_id": session_id,
            "ingested": session.get("ingested_at") is not None,
            "ingested_at": session.get("ingested_at"),
            "has_new_messages": has_new_messages,
            "chunk_count": chunk_count,
            "has_blob": has_blob,
            "message_count": session.get("message_count", 0)
        }
