"""
RAG System Orchestrator
Coordinates vector storage, retrieval, and generation components
"""

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
        
        # Initialize components
        self.gateway = AIGateway()
        self.vector_store = VectorStore(use_persistent=use_persistent if use_persistent is not None else self.config.rag_use_persistent)
        self.retriever = DocumentRetriever()
        
        # Setup collection
        self._setup_collection()
    
    def _setup_collection(self):
        """Setup the vector collection"""
        embedding_dim = self.retriever.get_embedding_dimension()
        success = self.vector_store.setup_collection(self.collection_name, embedding_dim)
        if not success:
            raise Exception(f"Failed to setup collection: {self.collection_name}")
    
    def add_documents(self, documents):
        """
        Add documents to the vector database
        
        Args:
            documents: List of text documents to index
            
        Returns:
            Number of documents added
        """
        # Create embeddings
        embeddings = self.retriever.encode_documents(documents)
        
        # Create points for vector store
        points = self.retriever.create_points(documents, embeddings)
        
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
            context_limit = self.config.top_k
            
        # Retrieve relevant documents
        retrieved_docs = self.search(question, limit=context_limit)
        
        if not retrieved_docs:
            return "No relevant documents found.", [], []
        
        # Build RAG context from retrieved documents
        rag_context = "\n\n".join([doc for doc, _ in retrieved_docs])
        
        # Create prompt
        prompt = f"""Context:
{rag_context}

Question: {question}

Answer:"""
        
        # Generate answer
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
