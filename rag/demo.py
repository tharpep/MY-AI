"""
Simple RAG Demo
Basic testing for RAG functionality
"""

import os
import time
from pathlib import Path
from .rag_setup import BasicRAG
from .document_ingester import DocumentIngester
from core.config import get_config
from core.utils.logging_config import log_rag_result


def run_rag_demo(mode="automated"):
    """Run RAG demo in automated mode"""
    print("=== RAG Demo ===")
    
    # Get configuration
    config = get_config()
    print(f"Using model: {config.model_name}")
    print(f"Provider: {'Ollama' if config.use_ollama else 'Purdue API'}")
    
    try:
        # Initialize RAG system (use in-memory for demo)
        print("\nInitializing RAG system...")
        print("Using in-memory storage for demo (no Docker required)")
        rag = BasicRAG(use_persistent=False)
        
        # Load documents
        print("Loading documents...")
        documents_folder = Path(__file__).parent.parent / "data" / "documents"
        
        if documents_folder.exists():
            # Check if we should clear the collection first
            if config.rag_clear_on_ingest:
                print("Clearing existing documents...")
                clear_result = rag.clear_collection()
                if clear_result.get("success"):
                    print(f"{clear_result['message']}")
                else:
                    print(f"Warning: {clear_result.get('error', 'Failed to clear collection')}")
            
            ingester = DocumentIngester(rag)
            result = ingester.ingest_folder(str(documents_folder))
            
            if result["success"]:
                print(f"Loaded {result['processed']} documents")
            else:
                print(f"Error loading documents: {result['error']}")
                return
        else:
            print(f"Documents folder not found: {documents_folder}")
            return
        
        # Show stats
        stats = rag.get_stats()
        print(f"\nCollection: {stats['collection_name']}")
        print(f"Documents: {stats['document_count']}")
        
        # Automated mode - run test queries
        print("\n=== Automated Mode ===")
        test_queries = [
            "What is Docker?",
            "How does binary search work?",
            "What is DevOps?",
            "Explain Python programming"
        ]
        
        for query in test_queries:
            print(f"\nQ: {query}")
            try:
                start_time = time.time()
                result = rag.query(query)
                response_time = time.time() - start_time
                
                # Handle new return format (answer, context_docs, context_scores)
                if isinstance(result, tuple):
                    answer, context_docs, context_scores = result
                else:
                    # Fallback for old format
                    answer = result
                    context_docs = []
                    context_scores = []
                
                print(f"A: {answer[:200]}...")  # Truncate long answers
                
                # Log the result
                log_rag_result(
                    question=query,
                    answer=answer,
                    response_time=response_time,
                    model_name=config.model_name,
                    provider="Ollama" if config.use_ollama else "Purdue API",
                    context_docs=context_docs,
                    context_scores=context_scores
                )
                
            except Exception as e:
                # Check if it's a timeout error
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    print(f"Timeout: Query took too long to process")
                else:
                    print(f"Error: {str(e)}")
        
        print("\nRAG demo completed successfully!")
        
    except Exception as e:
        print(f"Demo failed:")
        print(f"Raw error: {e}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")


def main():
    """Main function for direct execution"""
    run_rag_demo("automated")


if __name__ == "__main__":
    main()