"""Context Engine"""

from typing import List, Tuple, Optional
from llm.gateway import AIGateway
from .vector_store import VectorStore
from .retriever import DocumentRetriever
from .reranker import CrossEncoderReranker
from .query_processor import QueryProcessor
from .journal import JournalManager
from core.config import get_config

class ContextEngine:
    """Context Engine with dual-tier retrieval."""
    
    def __init__(self, collection_name=None, use_persistent=None, enable_journal=True):
        """Initialize Context Engine."""
        self.config = get_config()
        self.collection_name = collection_name or self.config.library_collection_name
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Initialized ContextEngine (ID: {id(self)})")
        
        self.gateway = AIGateway()
        self.vector_store = VectorStore(
            use_persistent=use_persistent if use_persistent is not None else self.config.storage_use_persistent,
            qdrant_host=self.config.qdrant_host,
            qdrant_port=self.config.qdrant_port
        )
        self.retriever = DocumentRetriever(model_name=self.config.embedding_model)
        
        self._journal: Optional[JournalManager] = None
        self._reranker: Optional[CrossEncoderReranker] = None
        self._query_processor: Optional[QueryProcessor] = None
        self._enable_journal = enable_journal
        
        self._setup_collection()
    
    def _setup_collection(self):
        try:
            from core.model_registry import get_configured_model
            model_info = get_configured_model("library")
            embedding_dim = model_info.dimension
            if not embedding_dim:
                embedding_dim = self.retriever.get_embedding_dimension()
        except Exception:
            embedding_dim = self.retriever.get_embedding_dimension()
            
        success = self.vector_store.setup_collection(self.collection_name, embedding_dim)
        if not success:
            raise Exception(f"Failed to setup collection: {self.collection_name}")
    
    @property
    def journal(self) -> Optional[JournalManager]:
        if self._enable_journal and self._journal is None:
            self._journal = JournalManager(
                vector_store=self.vector_store,
                embedder=self.retriever
            )
        return self._journal
    
    @property
    def reranker(self) -> Optional[CrossEncoderReranker]:
        if self.config.rerank_enabled and self._reranker is None:
            self._reranker = CrossEncoderReranker(model_name=self.config.rerank_model)
        return self._reranker
    
    @property
    def query_processor(self) -> Optional[QueryProcessor]:
        if self.config.query_expansion_enabled and self._query_processor is None:
            self._query_processor = QueryProcessor(
                gateway=self.gateway,
                model=self.config.query_expansion_model
            )
        return self._query_processor
    
    def add_documents(self, documents, metadata: dict = None):
        """Add documents to the vector database."""
        dense, sparse = self.retriever.encode_documents(documents)
        points = self.retriever.create_points(documents, dense, sparse, metadata=metadata)
        return self.vector_store.add_points(self.collection_name, points)
    
    def search(self, query, limit=None, expand_query=None):
        """Search for relevant documents with optional query expansion and reranking."""
        if limit is None:
            limit = self.config.chat_library_top_k
        
        search_query = query
        if expand_query is None:
            expand_query = self.config.query_expansion_enabled
        
        if expand_query and self.query_processor:
            search_query = self.query_processor.expand(query)
        
        if self.config.rerank_enabled:
            candidates = self.config.rerank_candidates
        else:
            candidates = limit
        
        dense, sparse = self.retriever.encode_query(search_query)
        results = self.vector_store.hybrid_search(
            self.collection_name, dense, sparse, candidates,
            sparse_weight=self.config.hybrid_sparse_weight
        )
        
        if self.config.rerank_enabled and results:
            docs = [doc for doc, _ in results]
            reranked = self.reranker.rerank(query, docs, top_k=limit)
            return reranked
        
        return results[:limit]
    
    def get_context_for_chat(
        self,
        query: str,
        top_k: int,
        similarity_threshold: float
    ) -> List[Tuple[str, float]]:
        """Get RAG context for chat endpoint."""
        retrieved = self.search(query, limit=top_k)
        
        filtered = [(doc, score) for doc, score in retrieved 
                   if score >= similarity_threshold]
        
        if self.config.log_output:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"RAG Retrieval - Query: '{query[:100]}...'")
            logger.info(f"  Top-K: {top_k}, Threshold: {similarity_threshold}")
            logger.info(f"  Retrieved: {len(retrieved)} docs, Filtered: {len(filtered)} docs")
            if retrieved:
                logger.info(f"  Retrieved Documents (before threshold filter):")
                for i, (doc, score) in enumerate(retrieved[:5], 1):  # Show top 5
                    doc_preview = doc[:150] + "..." if len(doc) > 150 else doc
                    passed = "✓" if score >= similarity_threshold else "✗"
                    logger.info(f"    [{i}] {passed} Score: {score:.3f} | {doc_preview}")
            if filtered:
                logger.info(f"  Documents passing threshold ({len(filtered)}):")
                for i, (doc, score) in enumerate(filtered, 1):
                    doc_preview = doc[:150] + "..." if len(doc) > 150 else doc
                    logger.info(f"    [{i}] Score: {score:.3f} | {doc_preview}")
            else:
                if retrieved:
                    max_score = max(score for _, score in retrieved) if retrieved else 0
                    logger.warning(f"  No documents passed threshold (max score: {max_score:.3f} < {similarity_threshold})")
                else:
                    logger.info(f"  No documents retrieved")
        
        return filtered
    
    def query(self, question, context_limit=None):
        """Answer a question using RAG."""
        if context_limit is None:
            context_limit = self.config.chat_library_top_k
            
        retrieved_docs = self.search(question, limit=context_limit)
        
        if self.config.log_output:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"RAG Query - Question: '{question[:100]}...'")
            logger.info(f"  Context Limit: {context_limit}")
            logger.info(f"  Retrieved: {len(retrieved_docs)} documents")
            if retrieved_docs:
                logger.info(f"  Retrieved Documents:")
                for i, (doc, score) in enumerate(retrieved_docs, 1):
                    doc_preview = doc[:150] + "..." if len(doc) > 150 else doc
                    logger.info(f"    [{i}] Score: {score:.3f} | {doc_preview}")
        
        if not retrieved_docs:
            return "No relevant documents found.", [], []
        
        rag_context = "\n\n".join([doc for doc, _ in retrieved_docs])
        
        from core.prompts import get_prompt, format_prompt
        rag_template = get_prompt("rag")
        prompt = format_prompt(rag_template, context=rag_context, question=question)
        
        if self.config.log_output:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"  Generating answer using model: {self.gateway.config.model_name}")
        
        answer = self.gateway.chat(prompt, provider=None, model=None)
        
        context_docs = [doc for doc, _ in retrieved_docs]
        context_scores = [score for _, score in retrieved_docs]
        
        return answer, context_docs, context_scores
    
    def get_stats(self):
        stats = self.vector_store.get_collection_stats(self.collection_name)
        if "error" not in stats:
            stats.update({
                "collection_name": stats.get("name", self.collection_name),
                "document_count": stats.get("points_count", 0),
                "vector_size": self.retriever.get_embedding_dimension(),
                "distance": "cosine",
                "model_info": self.retriever.get_model_info()
            })
        else:
            stats.update({
                "collection_name": self.collection_name,
                "document_count": 0,
                "vector_size": self.retriever.get_embedding_dimension(),
                "distance": "cosine",
                "model_info": self.retriever.get_model_info()
            })
        return stats
    
    def clear_collection(self):
        try:
            embedding_dim = self.retriever.get_embedding_dimension()
            
            self.vector_store.cleanup_old_collections([self.collection_name])
            
            self.vector_store.clear_collection(self.collection_name, embedding_dim)
            return {"success": True, "message": f"Cleared collection {self.collection_name}"}
        except Exception as e:
            return {"error": f"Failed to clear collection: {str(e)}"}
    
    def get_indexed_files(self) -> dict:
        """Get list of indexed files (blob_ids) with their chunk counts."""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue, ScrollRequest
            
            blob_counts = {}
            offset = None
            limit = 100
            
            while True:
                result = self.vector_store.client.scroll(
                    collection_name=self.collection_name,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                points, offset = result
                if not points:
                    break
                    
                for point in points:
                    blob_id = point.payload.get("blob_id")
                    filename = point.payload.get("original_filename", "unknown")
                    if blob_id:
                        if blob_id not in blob_counts:
                            blob_counts[blob_id] = {"blob_id": blob_id, "filename": filename, "chunk_count": 0}
                        blob_counts[blob_id]["chunk_count"] += 1
                
                if offset is None:
                    break
            
            return {
                "files": list(blob_counts.values()),
                "total_files": len(blob_counts)
            }
        except Exception as e:
            return {"error": f"Failed to get indexed files: {str(e)}", "files": [], "total_files": 0}
    
    def delete_by_blob_id(self, blob_id: str) -> dict:
        """Delete all chunks associated with a specific blob_id."""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            self.vector_store.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="blob_id",
                            match=MatchValue(value=blob_id)
                        )
                    ]
                )
            )
            
            return {"success": True, "message": f"Deleted all chunks for blob: {blob_id}"}
        except Exception as e:
            return {"error": f"Failed to delete chunks for {blob_id}: {str(e)}"}

_context_engine_instance: "ContextEngine | None" = None

def get_rag() -> "ContextEngine":
    global _context_engine_instance
    if _context_engine_instance is None:
        _context_engine_instance = ContextEngine()
    return _context_engine_instance


def main():
    pass


if __name__ == "__main__":
    main()
