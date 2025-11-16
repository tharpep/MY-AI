"""
Direct LLM Demo
Basic testing for direct LLM interaction without RAG
"""

import os
import time
from pathlib import Path
from .gateway import AIGateway
from core.config import get_config


def run_llm_demo(mode="automated"):
    """Run LLM demo in automated mode"""
    print("=== Direct LLM Demo ===")
    
    # Get configuration
    config = get_config()
    print(f"Using model: {config.model_name}")
    print(f"Provider: {'Ollama' if config.use_ollama else 'Purdue API'}")
    
    try:
        # Initialize LLM Gateway
        print("\nInitializing LLM Gateway...")
        gateway = AIGateway()
        
        # Test connection
        print("Testing connection...")
        test_response = gateway.chat("Hello, are you working?", provider=None, model=None)
        print(f"Connection test successful: {test_response[:50]}...")
        
        # Automated demo
        print("\n=== Automated Demo ===")
        
        demo_questions = [
            "Hello! How are you today?",
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about programming.",
            "What are the benefits of renewable energy?"
        ]
        
        for i, question in enumerate(demo_questions, 1):
            print(f"\n--- Question {i} ---")
            print(f"User: {question}")
            print("LLM: Thinking...")
            
            start_time = time.time()
            try:
                response = gateway.chat(question, provider=None, model=None)
                response_time = time.time() - start_time
                
                print(f"LLM: {response}")
                print(f"Response time: {response_time:.2f}s")
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                print(f"Full traceback:\n{traceback.format_exc()}")
            
            # Small delay between questions
            time.sleep(1)
        
        print("\n=== Automated Demo Complete ===")
            
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return 1
    
    return 0


async def run_embeddings_demo():
    """Run embeddings demo"""
    print("=== Embeddings Demo ===")
    
    try:
        # Initialize LLM Gateway
        print("Initializing LLM Gateway...")
        gateway = AIGateway()
        
        # Test embeddings
        test_texts = [
            "Hello, how are you?",
            "What is machine learning?",
            "The weather is nice today.",
            "I love programming in Python."
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n--- Embedding {i} ---")
            print(f"Text: {text}")
            
            try:
                start_time = time.time()
                embedding_response = await gateway.embeddings(text, model=None)
                response_time = time.time() - start_time
                
                embedding = embedding_response.get("embedding", [])
                print(f"Embedding dimension: {len(embedding)}")
                print(f"First 5 values: {embedding[:5]}")
                print(f"Response time: {response_time:.2f}s")
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                print(f"Full traceback:\n{traceback.format_exc()}")
        
        print("\n=== Embeddings Demo Complete ===")
        
    except Exception as e:
        print(f"Embeddings demo failed: {e}")
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    import asyncio
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "automated"
    
    if mode == "embeddings":
        result = asyncio.run(run_embeddings_demo())
    else:
        result = run_llm_demo(mode)
    
    sys.exit(result)
