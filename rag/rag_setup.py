"""
RAG System Orchestrator
Coordinates vector storage, retrieval, and generation components
"""

from typing import List, Tuple
from llm.gateway import AIGateway
from .vector_store import VectorStore
from .retriever import DocumentRetriever
from core.config import get_config


class BasicRAG:
    """RAG system that orchestrates vector storage, retrieval, and generation"""
    
    def __init__(self, collection_name=None, use_persistent=None):
        """
        Initialize RAG system
        
        Args:
            collection_name: Name for Qdrant collection (uses config default if None)
            use_persistent: If True, use persistent Qdrant storage (uses config default if None)
        """
        self.config = get_config()
        self.collection_name = collection_name or self.config.rag_collection_name
        
        # Initialize components with config values
        self.gateway = AIGateway()
        self.vector_store = VectorStore(
            use_persistent=use_persistent if use_persistent is not None else self.config.rag_use_persistent,
            qdrant_host=self.config.qdrant_host,
            qdrant_port=self.config.qdrant_port
        )
        self.retriever = DocumentRetriever(model_name=self.config.embedding_model)
        
        # Setup collection
        self._setup_collection()
    
    def _setup_collection(self):
        """Setup the vector collection"""
        embedding_dim = self.retriever.get_embedding_dimension()
        success = self.vector_store.setup_collection(self.collection_name, embedding_dim)
        if not success:
            raise Exception(f"Failed to setup collection: {self.collection_name}")
    
    def add_documents(self, documents, metadata: dict = None):
        """
        Add documents to the vector database
        
        Args:
            documents: List of text documents to index
            metadata: Optional metadata to store with each chunk (e.g., {"blob_id": "blob_xxx"})
            
        Returns:
            Number of documents added
        """
        # Create embeddings
        embeddings = self.retriever.encode_documents(documents)
        
        # Create points for vector store (includes metadata like blob_id)
        points = self.retriever.create_points(documents, embeddings, metadata=metadata)
        
        # Add to vector store
        return self.vector_store.add_points(self.collection_name, points)
    
    def search(self, query, limit=None):
        """
        Search for relevant documents
        
        Args:
            query: Search query
            limit: Number of results to return (uses config default if None)
            
        Returns:
            List of (text, score) tuples
        """
        # Use config default if limit not specified
        if limit is None:
            limit = self.config.rag_top_k
            
        # Create query embedding
        query_embedding = self.retriever.encode_query(query)
        
        # Search vector store
        return self.vector_store.search(self.collection_name, query_embedding, limit)
    
    def get_context_for_chat(
        self,
        query: str,
        top_k: int,
        similarity_threshold: float
    ) -> List[Tuple[str, float]]:
        """
        Get RAG context for chat endpoint.
        Performs vector search with top-k and filters by similarity threshold.

        Args:
            query: User query/message
            top_k: Number of documents to retrieve from vector search
            similarity_threshold: Minimum similarity score to include (0.0-1.0)

        Returns:
            List of (document_text, similarity_score) tuples that pass threshold.
            Empty list if no documents pass threshold.
        """
        # Perform vector search with top-k
        retrieved = self.search(query, limit=top_k)
        
        # Filter by similarity threshold
        filtered = [(doc, score) for doc, score in retrieved 
                   if score >= similarity_threshold]
        
        # Log retrieval details if logging enabled
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
        """
        Answer a question using RAG
        
        Args:
            question: Question to answer
            context_limit: Number of documents to retrieve for context (uses config default if None)
            
        Returns:
            Answer string
        """
        # Use config default if context_limit not specified
        if context_limit is None:
            context_limit = self.config.rag_top_k
            
        # Retrieve relevant documents
        retrieved_docs = self.search(question, limit=context_limit)
        
        # Log retrieval details if logging enabled
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
        
        # Build RAG context from retrieved documents
        rag_context = "\n\n".join([doc for doc, _ in retrieved_docs])
        
        # Create prompt from template
        from core.prompts import get_prompt, format_prompt
        rag_template = get_prompt("rag")
        prompt = format_prompt(rag_template, context=rag_context, question=question)
        
        # Generate answer
        if self.config.log_output:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"  Generating answer using model: {self.gateway.config.model_name}")
        
        answer = self.gateway.chat(prompt, provider=None, model=None)
        
        # Return answer along with context details for logging
        context_docs = [doc for doc, _ in retrieved_docs]
        context_scores = [score for _, score in retrieved_docs]
        
        return answer, context_docs, context_scores
    
    def get_stats(self):
        """Get collection statistics"""
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
            # If there's an error, still provide basic info
            stats.update({
                "collection_name": self.collection_name,
                "document_count": 0,
                "vector_size": self.retriever.get_embedding_dimension(),
                "distance": "cosine",
                "model_info": self.retriever.get_model_info()
            })
        return stats
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            embedding_dim = self.retriever.get_embedding_dimension()
            
            # Clean up old collections first
            self.vector_store.cleanup_old_collections([self.collection_name])
            
            # Clear the main collection
            self.vector_store.clear_collection(self.collection_name, embedding_dim)
            return {"success": True, "message": f"Cleared collection {self.collection_name}"}
        except Exception as e:
            return {"error": f"Failed to clear collection: {str(e)}"}
    
    def get_indexed_files(self) -> dict:
        """
        Get list of indexed files (blob_ids) with their chunk counts.
        
        Returns:
            Dictionary with list of indexed files and their stats
        """
        try:
            # Query all points with scroll to get unique blob_ids
            from qdrant_client.models import Filter, FieldCondition, MatchValue, ScrollRequest
            
            # Get all points (scrolling through)
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
        """
        Delete all chunks associated with a specific blob_id.
        
        Args:
            blob_id: The blob identifier to delete
            
        Returns:
            Dictionary with deletion result
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            # Delete all points with matching blob_id
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


# Singleton instance
_rag_instance: "BasicRAG | None" = None


def get_rag() -> BasicRAG:
    """
    Get the global BasicRAG instance (singleton pattern).
    
    Initializes once on first call, reuses on subsequent calls.
    This avoids repeated Qdrant connections and embedding model loads.
    
    Returns:
        BasicRAG instance
    """
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = BasicRAG()
    return _rag_instance


def main():
    """Demo of Basic RAG system"""
    print("=== Basic RAG Demo ===\n")
    
    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
        "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
        "Natural language processing (NLP) combines computational linguistics with machine learning to help computers understand human language.",
        "Computer vision enables machines to interpret and understand visual information from images and videos.",
        "Reinforcement learning is where agents learn optimal behavior by interacting with an environment and receiving rewards.",
    ]
    
    # Initialize RAG
    print("1. Initializing RAG system...")
    rag = BasicRAG()
    print(f"   Available providers: {rag.gateway.get_available_providers()}")
    
    # Add documents
    print("\n2. Adding documents...")
    count = rag.add_documents(documents)
    print(f"   Added {count} documents")
    
    # Show stats
    stats = rag.get_stats()
    print(f"   Collection stats: {stats}")
    
    # Test queries
    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "What is computer vision?"
    ]
    
    print("\n3. Testing queries...")
    for query in queries:
        print(f"\nQuery: {query}")
        
        # Get answer with RAG
        answer = rag.query(query)
        print(f"Answer: {answer}")
        
        # Show retrieved context
        retrieved = rag.search(query, limit=2)
        print(f"Retrieved docs: {len(retrieved)}")
        for i, (doc, score) in enumerate(retrieved, 1):
            print(f"  [{i}] (score: {score:.3f}) {doc[:100]}...")


if __name__ == "__main__":
    main()
