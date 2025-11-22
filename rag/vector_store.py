"""
Vector Store Operations
Handles Qdrant database operations for vector storage
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any, Tuple


class VectorStore:
    """Handles vector storage operations with Qdrant"""
    
    def __init__(self, use_persistent: bool = False):
        """
        Initialize vector store
        
        Args:
            use_persistent: If True, use persistent storage, otherwise in-memory
        """
        self.use_persistent = use_persistent
        
        # Setup Qdrant client
        if use_persistent:
            try:
                self.client = QdrantClient(path="./data/qdrant_db")
                print("Using persistent Qdrant storage")
            except Exception as e:
                # Handle corrupted database (e.g., old format incompatible with new client)
                error_str = str(e)
                error_lower = error_str.lower()
                is_corruption_error = (
                    "init_from" in error_str or 
                    "validation error" in error_lower or 
                    "extra_forbidden" in error_str or
                    "pydantic" in error_lower
                )
                
                if is_corruption_error:
                    import shutil
                    from pathlib import Path
                    import logging
                    
                    logger = logging.getLogger(__name__)
                    db_path = Path("./data/qdrant_db")
                    
                    logger.warning("Qdrant database appears corrupted (incompatible format). Attempting to fix...")
                    print("⚠️  Qdrant database format incompatible. Fixing automatically...")
                    
                    try:
                        # Backup old database
                        if db_path.exists():
                            backup_path = Path("./data/qdrant_db_backup")
                            if backup_path.exists():
                                shutil.rmtree(backup_path)
                            shutil.move(str(db_path), str(backup_path))
                            logger.info(f"Old database backed up to: {backup_path}")
                            print(f"   Backed up old database to: {backup_path}")
                        
                        # Recreate client with fresh database
                        self.client = QdrantClient(path="./data/qdrant_db")
                        logger.warning("Qdrant database recreated successfully.")
                        print("✅ Database recreated. You may need to re-ingest documents.")
                    except Exception as fix_error:
                        logger.error(f"Failed to fix database: {fix_error}")
                        logger.warning("Falling back to in-memory storage for this session.")
                        print(f"❌ Could not fix database: {fix_error}")
                        print("   Using in-memory storage for this session.")
                        self.client = QdrantClient(":memory:")
                        self.use_persistent = False  # Override to in-memory
                else:
                    # Other errors - re-raise with context
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Qdrant initialization failed: {e}")
                    raise RuntimeError(f"Failed to initialize Qdrant client: {e}") from e
        else:
            self.client = QdrantClient(":memory:")
            print("Using in-memory Qdrant storage")
    
    def setup_collection(self, collection_name: str, embedding_dim: int) -> bool:
        """
        Create Qdrant collection if it doesn't exist
        
        Args:
            collection_name: Name of the collection
            embedding_dim: Dimension of the vectors
            
        Returns:
            True if collection exists or was created successfully
        """
        try:
            # Check if collection exists
            self.client.get_collection(collection_name)
            return True
        except:
            # Create collection if it doesn't exist
            try:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dim, 
                        distance=Distance.COSINE
                    ),
                )
                return True
            except Exception as e:
                print(f"Error creating collection: {e}")
                return False
    
    def add_points(self, collection_name: str, points: List[PointStruct]) -> int:
        """
        Add points to the collection
        
        Args:
            collection_name: Name of the collection
            points: List of points to add
            
        Returns:
            Number of points added
        """
        try:
            self.client.upsert(collection_name=collection_name, points=points)
            return len(points)
        except Exception as e:
            print(f"Error adding points: {e}")
            return 0
    
    def search(self, collection_name: str, query_vector: List[float], limit: int = 3) -> List[Tuple[str, float]]:
        """
        Search for similar vectors
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector to search for
            limit: Maximum number of results
            
        Returns:
            List of (text, score) tuples
        """
        try:
            search_results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit
            ).points
            
            return [(hit.payload["text"], hit.score) for hit in search_results]
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get collection statistics
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection stats
        """
        try:
            collection_info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status
            }
        except Exception as e:
            return {"error": f"Could not retrieve collection info: {e}"}
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            self.client.delete_collection(collection_name)
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """
        List all collections
        
        Returns:
            List of collection names
        """
        try:
            collections = self.client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []
    
    def clear_collection(self, collection_name: str, embedding_dim: int = 384) -> bool:
        """
        Clear all points from a collection by recreating it
        
        Args:
            collection_name: Name of the collection to clear
            embedding_dim: Dimension of vectors (default: 384 for all-MiniLM-L6-v2)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete the entire collection
            try:
                self.client.delete_collection(collection_name)
            except:
                pass  # Collection might not exist
            
            # Recreate the collection immediately
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim, 
                    distance=Distance.COSINE
                ),
            )
            return True
        except Exception as e:
            print(f"Error clearing collection {collection_name}: {e}")
            return False
    
    def cleanup_old_collections(self, keep_collections: list = None):
        """
        Clean up old/unused collections
        
        Args:
            keep_collections: List of collection names to keep (default: ['simrag_docs'])
        """
        if keep_collections is None:
            keep_collections = ['simrag_docs']
        
        try:
            all_collections = self.list_collections()
            for collection_name in all_collections:
                if collection_name not in keep_collections:
                    try:
                        self.client.delete_collection(collection_name)
                    except Exception as e:
                        print(f"Could not delete collection {collection_name}: {e}")
        except Exception as e:
            print(f"Error during cleanup: {e}")