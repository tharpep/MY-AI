"""Vector Store Operations with Hybrid Search Support"""

import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVectorParams, SparseIndexParams
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


class VectorStore:
    """Handles vector storage operations with Qdrant"""
    
    def __init__(self, use_persistent: bool = False, qdrant_host: str = "localhost", qdrant_port: int = 6333):
        """Initialize vector store."""
        self.use_persistent = use_persistent
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        
        if use_persistent:
            try:
                self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
                self.client.get_collections()
                logger.info(f"Connected to Qdrant server at {qdrant_host}:{qdrant_port}")
            except Exception as e:
                error_str = str(e).lower()
                is_connection_error = any(term in error_str for term in ["connection", "refused", "timeout", "unreachable"])
                
                if is_connection_error:
                    logger.warning(f"Qdrant server not available at {qdrant_host}:{qdrant_port}: {e}")
                    logger.warning("Falling back to in-memory storage. Start Qdrant with: docker compose up qdrant -d")
                    self.client = QdrantClient(":memory:")
                    self.use_persistent = False
                else:
                    logger.error(f"Qdrant initialization failed: {e}")
                    raise RuntimeError(f"Failed to initialize Qdrant client: {e}") from e
        else:
            self.client = QdrantClient(":memory:")
            logger.info("Using in-memory Qdrant storage")
    
    def setup_collection(self, collection_name: str, embedding_dim: int) -> bool:
        """Create Qdrant collection if it doesn't exist."""
        try:
            self.client.get_collection(collection_name)
            return True
        except:
            try:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": VectorParams(size=embedding_dim, distance=Distance.COSINE)
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(index=SparseIndexParams())
                    },
                )
                return True
            except Exception as e:
                print(f"Error creating collection: {e}")
                return False
    
    def add_points(self, collection_name: str, points: List[PointStruct]) -> int:
        """Add points to the collection."""
        try:
            self.client.upsert(collection_name=collection_name, points=points)
            return len(points)
        except Exception as e:
            print(f"Error adding points: {e}")
            return 0
    
    def search(self, collection_name: str, query_vector: List[float], limit: int = 3) -> List[Tuple[str, float]]:
        """Search for similar vectors (dense only)."""
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
    
    def hybrid_search(self, collection_name: str, dense_vector: List[float], 
                     sparse_vector: Dict[int, float], limit: int = 3,
                     sparse_weight: float = 0.3) -> List[Tuple[str, float]]:
        """Perform hybrid search combining dense and sparse vectors with RRF fusion."""
        from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
        
        try:
            indices = list(sparse_vector.keys())
            values = list(sparse_vector.values())
            
            results = self.client.query_points(
                collection_name=collection_name,
                prefetch=[
                    Prefetch(query=dense_vector, using="dense", limit=limit * 2),
                    Prefetch(
                        query=SparseVector(indices=indices, values=values),
                        using="sparse",
                        limit=limit * 2
                    )
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit
            ).points
            
            return [(hit.payload["text"], hit.score) for hit in results]
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return self.search(collection_name, dense_vector, limit)
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics."""
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
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name)
            return True
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """List all collections."""
        try:
            collections = self.client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []
    
    def clear_collection(self, collection_name: str, embedding_dim: int = 1024) -> bool:
        """Clear all points from a collection by recreating it."""
        try:
            try:
                self.client.delete_collection(collection_name)
            except:
                pass
            
            return self.setup_collection(collection_name, embedding_dim)
        except Exception as e:
            print(f"Error clearing collection {collection_name}: {e}")
            return False
    
    def cleanup_old_collections(self, keep_collections: list = None):
        """Clean up old/unused collections."""
        if keep_collections is None:
            keep_collections = ['library_docs']
        
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