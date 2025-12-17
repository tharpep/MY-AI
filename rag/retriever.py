"""
Document Retriever
Handles embedding generation and document retrieval logic
"""

from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct
from typing import List, Tuple
import uuid


class DocumentRetriever:
    """Handles document embedding and retrieval operations"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize document retriever
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.retriever = SentenceTransformer(model_name)
        self.embedding_dim = self.retriever.get_sentence_embedding_dimension()
    
    def encode_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Encode documents into embeddings
        
        Args:
            documents: List of text documents
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.retriever.encode(documents)
        return [embedding.tolist() for embedding in embeddings]
    
    def encode_query(self, query: str) -> List[float]:
        """
        Encode a query into an embedding
        
        Args:
            query: Query text
            
        Returns:
            Query embedding vector
        """
        embedding = self.retriever.encode(query)
        return embedding.tolist()
    
    def create_points(self, documents: List[str], embeddings: List[List[float]], 
                     start_doc_id: int = 0, metadata: dict = None) -> List[PointStruct]:
        """
        Create Qdrant points from documents and embeddings
        
        Args:
            documents: List of text documents
            embeddings: List of embedding vectors
            start_doc_id: Starting document ID for indexing
            metadata: Optional metadata to include in each point (e.g., blob_id)
            
        Returns:
            List of Qdrant PointStruct objects
        """
        points = []
        for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
            payload = {
                "text": doc, 
                "doc_id": start_doc_id + idx,
                "chunk_id": idx
            }
            # Add any extra metadata (like blob_id)
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
        """
        Get the embedding dimension
        
        Returns:
            Embedding dimension size
        """
        return self.embedding_dim
    
    def get_model_info(self) -> dict:
        """
        Get information about the embedding model
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "max_seq_length": self.retriever.max_seq_length
        }
