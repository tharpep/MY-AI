"""Document Retriever"""

from qdrant_client.models import PointStruct
from typing import List, Tuple
import uuid


class DocumentRetriever:
    """Handles document embedding and retrieval operations"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """Initialize document retriever."""
        self.model_name = model_name
        self._retriever = None
        self._embedding_dim = None

    @property
    def retriever(self):
        if self._retriever is None:
            from sentence_transformers import SentenceTransformer
            self._retriever = SentenceTransformer(self.model_name)
        return self._retriever

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
             self._embedding_dim = self.retriever.get_sentence_embedding_dimension()
        return self._embedding_dim
    
    def encode_documents(self, documents: List[str]) -> List[List[float]]:
        """Encode documents into embeddings."""
        embeddings = self.retriever.encode(documents)
        return [embedding.tolist() for embedding in embeddings]
    
    def encode_query(self, query: str) -> List[float]:
        """Encode a query into an embedding."""
        embedding = self.retriever.encode(query)
        return embedding.tolist()
    
    def create_points(self, documents: List[str], embeddings: List[List[float]], 
                     start_doc_id: int = 0, metadata: dict = None) -> List[PointStruct]:
        """Create Qdrant points from documents and embeddings."""
        points = []
        for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
            payload = {
                "text": doc, 
                "doc_id": start_doc_id + idx,
                "chunk_id": idx
            }
            if metadata:
                payload.update(metadata)
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload
            )
            points.append(point)
        return points
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim
    
    def get_model_info(self) -> dict:
        """Get information about the embedding model."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "max_seq_length": self.retriever.max_seq_length
        }
