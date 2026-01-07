"""
Session Store for Journal Sessions
Persists session metadata and messages in SQLite for durable storage.

Sessions store chat history locally. Messages are saved in real-time,
then exported and ingested into RAG on trigger (manual or session switch).
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Database file path
DB_PATH = Path("./data/sessions.db")


class SessionStore:
    """
    Manages session metadata and messages in SQLite.

    Sessions are identified by session_id (UUID) which matches
    the session_id stored in Qdrant payloads after ingestion.

    Messages are stored in real-time during chat, separate from
    RAG ingestion which happens on trigger.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize session store."""
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create sessions and messages tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    message_count INTEGER DEFAULT 0,
                    ingested_at TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_activity
                ON sessions(last_activity DESC)
            """)

            # Migration: Add ingested_at column if it doesn't exist
            cursor.execute("PRAGMA table_info(sessions)")
            columns = [col[1] for col in cursor.fetchall()]
            if "ingested_at" not in columns:
                cursor.execute("ALTER TABLE sessions ADD COLUMN ingested_at TEXT")
                logger.info("Migrated sessions table: added ingested_at column")

            # Messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp
                ON messages(session_id, timestamp)
            """)

            conn.commit()
            logger.info(f"Session store initialized at {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper cleanup."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Session store error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def upsert_session(self, session_id: str, name: Optional[str] = None) -> None:
        """
        Create or update a session.
        
        Args:
            session_id: Unique session identifier
            name: Optional friendly name for the session
        """
        now = datetime.utcnow().isoformat()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if session exists
            cursor.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
            exists = cursor.fetchone() is not None
            
            if exists:
                # Update last_activity
                cursor.execute("""
                    UPDATE sessions SET last_activity = ? WHERE session_id = ?
                """, (now, session_id))
                if name:
                    cursor.execute("""
                        UPDATE sessions SET name = ? WHERE session_id = ?
                    """, (name, session_id))
            else:
                # Insert new session
                cursor.execute("""
                    INSERT INTO sessions (session_id, name, created_at, last_activity, message_count)
                    VALUES (?, ?, ?, ?, 0)
                """, (session_id, name, now, now))
            
            conn.commit()
    
    def increment_message_count(self, session_id: str) -> None:
        """Increment the message count for a session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions SET message_count = message_count + 1 
                WHERE session_id = ?
            """, (session_id,))
            conn.commit()
    
    def set_session_name(self, session_id: str, name: str) -> None:
        """Set or update the friendly name for a session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions SET name = ? WHERE session_id = ?
            """, (name, session_id))
            conn.commit()
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a single session by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List sessions ordered by last activity (most recent first)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM sessions 
                ORDER BY last_activity DESC 
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages. Returns True if deleted."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Delete messages first (foreign key)
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            # Delete session
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0

    # =========================================================================
    # Message Methods
    # =========================================================================

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        timestamp: Optional[str] = None
    ) -> int:
        """
        Add a chat message to a session.

        Args:
            session_id: Session identifier
            role: Message role ('user' or 'assistant')
            content: Message content
            timestamp: Optional timestamp (defaults to now)

        Returns:
            The message ID
        """
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO messages (session_id, role, content, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, timestamp),
            )
            message_id = cursor.lastrowid
            conn.commit()

        return message_id

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all messages for a session, ordered by timestamp.

        Args:
            session_id: Session identifier

        Returns:
            List of message dicts with role, content, timestamp
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, role, content, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (session_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_session_with_messages(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a session with all its messages.

        Args:
            session_id: Session identifier

        Returns:
            Session dict with 'messages' list, or None if not found
        """
        session = self.get_session(session_id)
        if session is None:
            return None

        session["messages"] = self.get_messages(session_id)
        return session

    def get_first_user_message(self, session_id: str) -> Optional[str]:
        """
        Get the first user message content for a session.

        Used for auto-generating session titles.

        Args:
            session_id: Session identifier

        Returns:
            First user message content, or None if no user messages
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT content FROM messages
                WHERE session_id = ? AND role = 'user'
                ORDER BY timestamp ASC, id ASC
                LIMIT 1
                """,
                (session_id,),
            )
            row = cursor.fetchone()
            return row["content"] if row else None

    def delete_messages(self, session_id: str) -> int:
        """
        Delete all messages for a session (keeps session metadata).

        Args:
            session_id: Session identifier

        Returns:
            Number of messages deleted
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
            return cursor.rowcount

    # =========================================================================
    # Ingestion Tracking Methods
    # =========================================================================

    def set_ingested_at(self, session_id: str, timestamp: Optional[str] = None) -> None:
        """
        Mark a session as ingested into RAG.

        Args:
            session_id: Session identifier
            timestamp: Ingestion timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE sessions SET ingested_at = ? WHERE session_id = ?",
                (timestamp, session_id),
            )
            conn.commit()

    def clear_ingested_at(self, session_id: str) -> None:
        """
        Clear the ingested_at timestamp (mark as not ingested).

        Args:
            session_id: Session identifier
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE sessions SET ingested_at = NULL WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()

    def has_new_messages_since_ingest(self, session_id: str) -> bool:
        """
        Check if a session has new messages since last ingestion.

        Returns True if:
        - Session has never been ingested (ingested_at is NULL), OR
        - Session's last_activity is after ingested_at

        Args:
            session_id: Session identifier

        Returns:
            True if session needs (re)ingestion
        """
        session = self.get_session(session_id)
        if session is None:
            return False

        # No messages = nothing to ingest
        if session.get("message_count", 0) == 0:
            return False

        ingested_at = session.get("ingested_at")

        # Never ingested but has messages
        if ingested_at is None:
            return True

        # Compare timestamps
        last_activity = session.get("last_activity")
        if last_activity is None:
            return False

        return last_activity > ingested_at

    def get_sessions_needing_ingest(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get sessions that have messages but haven't been ingested or have new content.

        Useful for batch ingestion or startup checks.

        Args:
            limit: Maximum sessions to return

        Returns:
            List of session dicts needing ingestion
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM sessions
                WHERE message_count > 0
                  AND (ingested_at IS NULL OR last_activity > ingested_at)
                ORDER BY last_activity DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]


# Singleton instance
_session_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    """Get the global SessionStore instance."""
    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
    return _session_store
