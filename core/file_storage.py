"""
Blob Storage for Pre-Index Files
Manages file uploads in data/preindex_blob/ for RAG ingestion pipeline
"""

import uuid
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, asdict


# Default blob storage directory
BLOB_STORAGE_PATH = Path("./data/preindex_blob")


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
    """Get the global BlobStorage instance"""
    global _blob_storage
    if _blob_storage is None:
        _blob_storage = BlobStorage()
    return _blob_storage
