"""Document Retriever with Hybrid Embedding"""

from qdrant_client.models import PointStruct, SparseVector
from typing import List, Tuple, Dict
import uuid


class DocumentRetriever:
    """Handles document embedding with BGE-M3 dense+sparse vectors."""
    
    def __init__(self, model_name: str = 'BAAI/bge-m3'):
        """Initialize document retriever."""
        self.model_name = model_name
        self._flag_model = None
        self._embedding_dim = 1024  # BGE-M3 dense dimension

    @property
    def flag_model(self):
        """Get FlagEmbedding model for hybrid embeddings."""
        if self._flag_model is None:
            from FlagEmbedding import BGEM3FlagModel
            import torch
            use_fp16 = torch.cuda.is_available()
            self._flag_model = BGEM3FlagModel(self.model_name, use_fp16=use_fp16)
        return self._flag_model

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    def encode_documents(self, documents: List[str]) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        """Encode documents into dense and sparse embeddings."""
        output = self.flag_model.encode(documents, return_dense=True, return_sparse=True)
        dense = [emb.tolist() for emb in output['dense_vecs']]
        sparse = self._convert_sparse(output['lexical_weights'])
        return dense, sparse
    
    def _convert_sparse(self, lexical_weights: List[Dict]) -> List[Dict[int, float]]:
        """Convert FlagEmbedding sparse output to Qdrant format."""
        result = []
        for weights in lexical_weights:
            sparse_vec = {int(k): float(v) for k, v in weights.items()}
            result.append(sparse_vec)
        return result
    
    def encode_query(self, query: str) -> Tuple[List[float], Dict[int, float]]:
        """Encode query into dense and sparse embeddings."""
        output = self.flag_model.encode([query], return_dense=True, return_sparse=True)
        dense = output['dense_vecs'][0].tolist()
        sparse = self._convert_sparse(output['lexical_weights'])[0]
        return dense, sparse
    
    def create_points(self, documents: List[str], dense_embeddings: List[List[float]],
                     sparse_embeddings: List[Dict[int, float]], start_doc_id: int = 0,
                     metadata: dict = None) -> List[PointStruct]:
        """Create Qdrant points with enriched metadata."""
        from datetime import datetime, timezone
        
        points = []
        ingested_at = datetime.now(timezone.utc).isoformat()
        
        for idx, (doc, dense, sparse) in enumerate(zip(documents, dense_embeddings, sparse_embeddings)):
            payload = {
                "text": doc, 
                "doc_id": start_doc_id + idx,
                "chunk_id": idx,
                "ingested_at": ingested_at
            }
            
            if metadata:
                if "document_type" in metadata:
                    payload["document_type"] = metadata["document_type"]
                if "tags" in metadata:
                    payload["tags"] = metadata["tags"]
                if "section_title" in metadata:
                    payload["section_title"] = metadata["section_title"]
                if "source_file" in metadata:
                    payload["source_file"] = metadata["source_file"]
                for k, v in metadata.items():
                    if k not in payload:
                        payload[k] = v
            
            indices = list(sparse.keys())
            values = list(sparse.values())
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense,
                    "sparse": SparseVector(indices=indices, values=values)
                },
                payload=payload
            )
            points.append(point)
        return points
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self._embedding_dim
    
    def get_model_info(self) -> dict:
        """Get information about the embedding model."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self._embedding_dim,
            "hybrid": True
        }
