"""Journal Manager"""

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
    """Schema for a journal entry stored in Qdrant payload."""
    role: Literal["user", "assistant"]
    content: str
    session_id: str
    timestamp: str


class JournalChunkPayload(BaseModel):
    """Schema for a journal chunk stored in Qdrant."""
    text: str
    session_id: str
    session_name: Optional[str]
    chunk_index: int
    total_chunks: int
    message_count: int
    ingested_at: str


class JournalManager:
    """Manages chat history ingestion and retrieval."""

    def __init__(self, vector_store: Optional[VectorStore] = None, embedder=None):
        """Initialize the Journal Manager."""
        self.config = get_config()

        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = VectorStore(
                use_persistent=self.config.storage_use_persistent,
                qdrant_host=self.config.qdrant_host,
                qdrant_port=self.config.qdrant_port
            )

        self.model_info = get_configured_model("journal")
        self.embedding_dim = self.model_info.dimension or 384

        self._shared_embedder = embedder
        self._embedder = None

        self._setup_collection()

        logger.info(f"JournalManager initialized with model: {self.model_info.name}")

    def _setup_collection(self) -> None:
        self.vector_store.setup_collection(
            collection_name=self.config.journal_collection_name,
            embedding_dim=self.embedding_dim
        )

    @property
    def embedder(self):
        """Get the embedding model (shared or lazy-loaded)."""
        if self._shared_embedder is not None:
            return self._shared_embedder
        
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.model_info.name)
            logger.info(f"Loaded embedding model: {self.model_info.name}")
        return self._embedder
    
    def _encode(self, text: str) -> List[float]:
        """Encode text to embedding, handling both embedder types."""
        if self._shared_embedder is not None:
            if hasattr(self._shared_embedder, 'encode_documents'):
                return self._shared_embedder.encode_documents([text])[0]
            return self._shared_embedder.encode(text).tolist()
        
        return self.embedder.encode(text).tolist()

    def ingest_session(self, session_id: str) -> Dict[str, Any]:
        """Ingest a session into the journal RAG collection."""
        from core.session_store import get_session_store
        from core.file_storage import get_journal_blob_storage

        logger.info(f"Starting ingestion for session: {session_id}")

        session_store = get_session_store()
        session_data = session_store.get_session_with_messages(session_id)

        if session_data is None:
            logger.error(f"Session not found: {session_id}")
            return {"error": f"Session not found: {session_id}"}

        messages = session_data.get("messages", [])
        if not messages:
            logger.warning(f"Session has no messages: {session_id}")
            return {"error": "Session has no messages", "session_id": session_id}

        blob_storage = get_journal_blob_storage()
        blob_path = blob_storage.export_session(session_id, session_data)
        logger.info(f"Exported session to: {blob_path}")

        deleted_count = self.delete_session_chunks(session_id)
        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} existing chunks for session {session_id}")

        conversation_text = self._format_conversation_for_ingestion(session_data)
        
        from rag.chunking import chunk_conversation
        chunks = chunk_conversation(
            conversation_text,
            chunk_size=self.config.journal_chunk_size,
            overlap=self.config.journal_chunk_overlap
        )
        logger.info(f"Created {len(chunks)} chunks from {len(messages)} messages")

        ingested_at = datetime.utcnow().isoformat()
        points = []

        for i, chunk_text in enumerate(chunks):
            embedding = self._encode(chunk_text)

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

        chunks_created = self.vector_store.add_points(
            self.config.journal_collection_name,
            points
        )
        logger.info(f"Stored {chunks_created} chunks in Qdrant")

        session_store.set_ingested_at(session_id, ingested_at)

        return {
            "session_id": session_id,
            "chunks_created": chunks_created,
            "blob_path": blob_path,
            "ingested_at": ingested_at,
            "message_count": len(messages)
        }

    def _format_conversation_for_ingestion(self, session_data: Dict[str, Any]) -> str:
        parts = []

        if session_data.get("name"):
            parts.append(f"Conversation: {session_data['name']}")
            parts.append("")

        for msg in session_data.get("messages", []):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            parts.append(f"[{role}] {content}")

        return "\n\n".join(parts)

    def delete_session_chunks(self, session_id: str) -> int:
        """Delete all chunks for a session from Qdrant."""
        try:
            count_before = self.get_session_chunk_count(session_id)

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
        """Count chunks in Qdrant for a session."""
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

    def get_context_for_chat(
        self,
        query: str,
        top_k: int,
        similarity_threshold: float,
        session_id: Optional[str] = None
    ) -> List[tuple[str, float]]:
        """Get RAG context for chat endpoint."""
        try:
            query_vector = self._encode(query)
            
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
            
            results = self.vector_store.client.query_points(
                collection_name=self.config.journal_collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k
            ).points
            
            retrieved = [(hit.payload.get("text", ""), hit.score) for hit in results]
            
            filtered = [(text, score) for text, score in retrieved if score >= similarity_threshold]
            
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
        """Retrieve relevant chat history for context."""
        try:
            query_vector = self._encode(query)

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

            results = self.vector_store.client.query_points(
                collection_name=self.config.journal_collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit
            ).points

            entries = []
            for hit in results:
                try:
                    payload = hit.payload

                    if "text" in payload:
                        entry = JournalEntry(
                            role="assistant",
                            content=payload.get("text", ""),
                            session_id=payload.get("session_id", ""),
                            timestamp=payload.get("ingested_at", "")
                        )
                    else:
                        entry = JournalEntry(**payload)

                    entries.append(entry)
                except Exception as e:
                    logger.warning(f"Failed to parse journal entry: {e}")

            return entries

        except Exception as e:
            logger.error(f"Failed to retrieve journal context: {e}")
            return []

    async def delete_session(self, session_id: str) -> bool:
        """Delete all data for a session."""
        from core.session_store import get_session_store
        from core.file_storage import get_journal_blob_storage

        try:
            self.delete_session_chunks(session_id)

            blob_storage = get_journal_blob_storage()
            blob_storage.delete_session(session_id)

            session_store = get_session_store()
            session_store.delete_session(session_id)

            logger.info(f"Deleted all data for session: {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False

    def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List sessions from SQLite."""
        from core.session_store import get_session_store

        try:
            session_store = get_session_store()
            return session_store.list_sessions(limit=limit)
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    async def clear_all(self) -> bool:
        """Clear all entries from the journal Qdrant collection."""
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
        """Get ingestion status for a session."""
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
