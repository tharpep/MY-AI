"""
Blob Storage for Pre-Index Files and Journal Sessions

Manages:
- Library blob storage: file uploads in data/library_blob/ for document RAG
- Journal blob storage: exported sessions in data/journal_blob/ for chat history RAG
"""

import uuid
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Default blob storage directories
BLOB_STORAGE_PATH = Path("./data/preindex_blob")
JOURNAL_BLOB_STORAGE_PATH = Path("./data/journal_blob")


@dataclass
class BlobInfo:
    """Information about a stored blob"""
    blob_id: str
    original_filename: str
    file_extension: str
    size_bytes: int
    created_at: str
    storage_path: str


class BlobStorage:
    """
    Manages file storage in the preindex_blob directory.
    
    Files are stored with unique IDs to prevent collisions.
    Original filenames and metadata are preserved.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize blob storage.
        
        Args:
            storage_path: Custom storage path. Defaults to ./data/preindex_blob/
        """
        self.storage_path = Path(storage_path) if storage_path else BLOB_STORAGE_PATH
        self._ensure_storage_exists()
    
    def _ensure_storage_exists(self) -> None:
        """Create storage directory if it doesn't exist"""
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def _generate_blob_id(self) -> str:
        """Generate a unique blob ID"""
        return f"blob_{uuid.uuid4().hex[:12]}"
    
    def _get_manifest_path(self) -> Path:
        """Get path to the blob manifest file"""
        return self.storage_path / "_manifest.json"
    
    def _load_manifest(self) -> dict:
        """Load blob manifest from disk"""
        manifest_path = self._get_manifest_path()
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_manifest(self, manifest: dict) -> None:
        """Save blob manifest to disk"""
        manifest_path = self._get_manifest_path()
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
    
    def save(self, file_content: bytes, original_filename: str) -> str:
        """
        Save a file to blob storage.
        
        Args:
            file_content: Raw file bytes
            original_filename: Original name of the file
            
        Returns:
            blob_id: Unique identifier for the stored blob
        """
        blob_id = self._generate_blob_id()
        file_extension = Path(original_filename).suffix.lower()
        
        # Create storage filename: blob_id + original extension
        storage_filename = f"{blob_id}{file_extension}"
        storage_path = self.storage_path / storage_filename
        
        # Write file
        with open(storage_path, 'wb') as f:
            f.write(file_content)
        
        # Create blob info
        blob_info = BlobInfo(
            blob_id=blob_id,
            original_filename=original_filename,
            file_extension=file_extension,
            size_bytes=len(file_content),
            created_at=datetime.utcnow().isoformat(),
            storage_path=str(storage_path)
        )
        
        # Update manifest
        manifest = self._load_manifest()
        manifest[blob_id] = asdict(blob_info)
        self._save_manifest(manifest)
        
        return blob_id
    
    def get(self, blob_id: str) -> Optional[Path]:
        """
        Get the file path for a blob.
        
        Args:
            blob_id: The blob identifier
            
        Returns:
            Path to the file, or None if not found
        """
        manifest = self._load_manifest()
        if blob_id not in manifest:
            return None
        
        storage_path = Path(manifest[blob_id]["storage_path"])
        if not storage_path.exists():
            return None
        
        return storage_path
    
    def get_info(self, blob_id: str) -> Optional[BlobInfo]:
        """
        Get full blob info including original filename.
        
        Args:
            blob_id: The blob identifier
            
        Returns:
            BlobInfo with original filename, or None if not found
        """
        manifest = self._load_manifest()
        if blob_id not in manifest:
            return None
        
        return BlobInfo(**manifest[blob_id])
    

    def list(self) -> List[BlobInfo]:
        """
        List all blobs in storage.
        
        Returns:
            List of BlobInfo objects
        """
        manifest = self._load_manifest()
        return [BlobInfo(**info) for info in manifest.values()]
    
    def delete(self, blob_id: str) -> bool:
        """
        Delete a blob from storage.
        
        Args:
            blob_id: The blob identifier
            
        Returns:
            True if deleted, False if not found
        """
        manifest = self._load_manifest()
        if blob_id not in manifest:
            return False
        
        # Delete file
        storage_path = Path(manifest[blob_id]["storage_path"])
        if storage_path.exists():
            storage_path.unlink()
        
        # Remove from manifest
        del manifest[blob_id]
        self._save_manifest(manifest)
        
        return True
    


# Singleton instance for convenience
_blob_storage: Optional[BlobStorage] = None


def get_blob_storage() -> BlobStorage:
    """Get the global BlobStorage instance (uses config for path)"""
    global _blob_storage
    if _blob_storage is None:
        from core.config import get_config
        config = get_config()
        _blob_storage = BlobStorage(storage_path=Path(config.blob_storage_path))
    return _blob_storage


# =============================================================================
# Journal Blob Storage - For exported chat sessions
# =============================================================================


@dataclass
class JournalBlobInfo:
    """Information about an exported journal session"""
    session_id: str
    name: Optional[str]
    message_count: int
    exported_at: str
    storage_path: str


class JournalBlobStorage:
    """
    Manages exported journal sessions as JSON files.

    Each session is stored as {session_id}.json containing:
    - Session metadata (name, timestamps)
    - Full message history

    Used for RAG ingestion of chat history.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize journal blob storage.

        Args:
            storage_path: Custom storage path. Defaults to ./data/journal_blob/
        """
        self.storage_path = Path(storage_path) if storage_path else JOURNAL_BLOB_STORAGE_PATH
        self._ensure_storage_exists()

    def _ensure_storage_exists(self) -> None:
        """Create storage directory if it doesn't exist"""
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, session_id: str) -> Path:
        """Get the file path for a session"""
        return self.storage_path / f"{session_id}.json"

    def export_session(self, session_id: str, session_data: Dict[str, Any]) -> str:
        """
        Export a session to blob storage as JSON.

        Args:
            session_id: Session identifier
            session_data: Session dict containing:
                - name: Session name
                - created_at: Creation timestamp
                - messages: List of message dicts

        Returns:
            Path to the exported file
        """
        export_data = {
            "session_id": session_id,
            "name": session_data.get("name"),
            "created_at": session_data.get("created_at"),
            "exported_at": datetime.utcnow().isoformat(),
            "message_count": len(session_data.get("messages", [])),
            "messages": session_data.get("messages", [])
        }

        file_path = self._get_session_path(session_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported session {session_id} to {file_path}")
        return str(file_path)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load an exported session from blob storage.

        Args:
            session_id: Session identifier

        Returns:
            Session data dict, or None if not found
        """
        file_path = self._get_session_path(session_id)
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def exists(self, session_id: str) -> bool:
        """
        Check if a session export exists.

        Args:
            session_id: Session identifier

        Returns:
            True if session export exists
        """
        return self._get_session_path(session_id).exists()

    def delete_session(self, session_id: str) -> bool:
        """
        Delete an exported session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        file_path = self._get_session_path(session_id)
        if not file_path.exists():
            return False

        file_path.unlink()
        logger.info(f"Deleted session export {session_id}")
        return True

    def list_sessions(self) -> List[JournalBlobInfo]:
        """
        List all exported sessions.

        Returns:
            List of JournalBlobInfo objects
        """
        sessions = []
        for file_path in self.storage_path.glob("*.json"):
            if file_path.name.startswith("_"):
                continue  # Skip manifest files

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                sessions.append(JournalBlobInfo(
                    session_id=data.get("session_id", file_path.stem),
                    name=data.get("name"),
                    message_count=data.get("message_count", 0),
                    exported_at=data.get("exported_at", ""),
                    storage_path=str(file_path)
                ))
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                continue

        # Sort by exported_at descending
        sessions.sort(key=lambda s: s.exported_at or "", reverse=True)
        return sessions

    def get_session_text(self, session_id: str) -> Optional[str]:
        """
        Get session content as plain text for RAG ingestion.

        Formats messages as:
        [USER] message content
        [ASSISTANT] response content

        Args:
            session_id: Session identifier

        Returns:
            Formatted text, or None if session not found
        """
        session_data = self.get_session(session_id)
        if session_data is None:
            return None

        parts = []

        # Add session context header
        if session_data.get("name"):
            parts.append(f"Session: {session_data['name']}")
            parts.append("")

        # Format messages
        for msg in session_data.get("messages", []):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            parts.append(f"[{role}] {content}")

        return "\n\n".join(parts)


# Singleton instance for journal blob storage
_journal_blob_storage: Optional[JournalBlobStorage] = None


def get_journal_blob_storage() -> JournalBlobStorage:
    """Get the global JournalBlobStorage instance (uses config for path)"""
    global _journal_blob_storage
    if _journal_blob_storage is None:
        from core.config import get_config
        config = get_config()
        _journal_blob_storage = JournalBlobStorage(
            storage_path=Path(config.journal_blob_storage_path)
        )
    return _journal_blob_storage
