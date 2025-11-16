"""
Tests for all API endpoints including health, LLM, and RAG functionality
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app
from core.config import get_config

# Fixture for the event loop, required by pytest-asyncio
@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for each test module."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def client():
    """Create a test client for testing FastAPI endpoints."""
    return TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_health_check(self, client: TestClient):
        """Test basic health check endpoint"""
        response = client.get("/health/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "personal-ai-assistant-api"
        assert data["version"] == "0.1.0"
    
    def test_detailed_health_check(self, client: TestClient):
        """Test detailed health check endpoint"""
        response = client.get("/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["service"] == "personal-ai-assistant-api"
        assert data["version"] == "0.1.0"
        # Status can be healthy, degraded, or unhealthy depending on gateway status


class TestLLMEndpoints:
    """Test LLM-related endpoints"""
    
    def test_list_models(self, client: TestClient):
        """Test models listing endpoint"""
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert isinstance(data["data"], list)
        # Should have at least one model
        assert len(data["data"]) > 0
    
    def test_chat_completions(self, client: TestClient):
        """Test chat completions endpoint"""
        config = get_config()
        model_name = config.model_name
        
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Hello, how are you?"}]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]
    
    def test_chat_completions_invalid_model(self, client: TestClient):
        """Test chat completions with invalid model"""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "invalid-model",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        # Should still work but might use default model
        assert response.status_code in [200, 400]
    
    def test_embeddings(self, client: TestClient):
        """Test embeddings endpoint"""
        response = client.post(
            "/v1/embeddings",
            json={
                "input": "This is a test sentence for embeddings."
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0
        assert "embedding" in data["data"][0]
        assert isinstance(data["data"][0]["embedding"], list)
    
    def test_embeddings_invalid_input(self, client: TestClient):
        """Test embeddings with invalid input"""
        response = client.post(
            "/v1/embeddings",
            json={
                "input": ""  # Empty input
            }
        )
        assert response.status_code in [200, 400]


class TestRAGEndpoints:
    """Test RAG-related endpoints"""
    
    def test_rag_query(self, client: TestClient):
        """Test RAG query endpoint"""
        response = client.post(
            "/v1/query",
            json={
                "question": "What is this system about?",
                "summarize": False
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert isinstance(data["sources"], list)
    
    def test_rag_query_with_summarize(self, client: TestClient):
        """Test RAG query with summarization"""
        response = client.post(
            "/v1/query",
            json={
                "question": "What is this system about?",
                "summarize": True
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
    
    def test_rag_query_invalid_question(self, client: TestClient):
        """Test RAG query with invalid question"""
        response = client.post(
            "/v1/query",
            json={
                "question": "",  # Empty question
                "summarize": False
            }
        )
        assert response.status_code in [200, 400]
    
    def test_ingest_documents(self, client: TestClient):
        """Test document ingestion endpoint"""
        response = client.post(
            "/v1/ingest",
            json={"folder_path": "./data/documents"}
        )
        # Should work if documents exist, or return appropriate error
        assert response.status_code in [200, 400, 404]
        if response.status_code == 200:
            data = response.json()
            assert "message" in data
            assert "processed" in data
    
    def test_ingest_documents_invalid_path(self, client: TestClient):
        """Test document ingestion with invalid path"""
        response = client.post(
            "/v1/ingest",
            json={"folder_path": "/nonexistent/path"}
        )
        assert response.status_code in [400, 404]
    
    def test_rag_stats(self, client: TestClient):
        """Test RAG statistics endpoint"""
        response = client.get("/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert "collections" in data
        assert "total_documents" in data
        assert "total_chunks" in data


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_json(self, client: TestClient):
        """Test handling of invalid JSON"""
        response = client.post(
            "/v1/chat/completions",
            content="invalid json"
        )
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_missing_required_fields(self, client: TestClient):
        """Test handling of missing required fields"""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model"
                # Missing "messages" field
            }
        )
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_nonexistent_endpoint(self, client: TestClient):
        """Test handling of nonexistent endpoints"""
        response = client.get("/v1/nonexistent")
        assert response.status_code == 404


class TestIntegration:
    """Test integration scenarios"""
    
    def test_full_rag_workflow(self, client: TestClient):
        """Test complete RAG workflow: ingest -> query"""
        # First, ingest documents
        ingest_response = client.post(
            "/v1/ingest",
            json={"folder_path": "./data/documents"}
        )
        
        # Then query
        query_response = client.post(
            "/v1/query",
            json={
                "question": "What is this system about?",
                "summarize": False
            }
        )
        
        # Both should work (or fail gracefully)
        assert ingest_response.status_code in [200, 400, 404]
        assert query_response.status_code == 200
    
    def test_health_to_chat_workflow(self, client: TestClient):
        """Test workflow from health check to chat completion"""
        # Check health first
        health_response = client.get("/health/")
        assert health_response.status_code == 200
        
        # Then chat
        chat_response = client.post(
            "/v1/chat/completions",
            json={
                "model": get_config().model_name,
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        assert chat_response.status_code == 200