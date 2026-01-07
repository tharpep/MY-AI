"""Minimal Ollama client wrapper for local development."""

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from .base_client import BaseLLMClient
from core.config import get_config


DEFAULT_MODEL = "llama3.2:1b"


@dataclass
class OllamaConfig:
    base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    default_model: str = field(default_factory=lambda: get_config().model_ollama)
    chat_timeout: float = field(default_factory=lambda: float(os.getenv("OLLAMA_CHAT_TIMEOUT", "60.0")))
    embeddings_timeout: float = field(default_factory=lambda: float(os.getenv("OLLAMA_EMBEDDINGS_TIMEOUT", "30.0")))
    connection_timeout: float = field(default_factory=lambda: float(os.getenv("OLLAMA_CONNECTION_TIMEOUT", "5.0")))


class OllamaClient(BaseLLMClient):
    """Very small Ollama HTTP client."""

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()
        self.logger = logging.getLogger(__name__)
        
        self._sync_client: Optional[httpx.Client] = None
        self._async_client: Optional[httpx.AsyncClient] = None
        
        if not self._check_ollama_health():
            raise ConnectionError(f"Ollama is not running or not accessible at {self.config.base_url}. Please start Ollama with 'ollama serve'")

    async def __aenter__(self):
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
    
    def __del__(self):
        if hasattr(self, '_sync_client') and self._sync_client:
            self._sync_client.close()

    def _ensure_sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(
                    connect=self.config.connection_timeout,
                    read=max(self.config.chat_timeout, self.config.embeddings_timeout),
                    write=self.config.connection_timeout,
                    pool=self.config.connection_timeout,
                ),
            )
        return self._sync_client

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(
                    connect=self.config.connection_timeout,
                    read=max(self.config.chat_timeout, self.config.embeddings_timeout),
                    write=self.config.connection_timeout,
                    pool=self.config.connection_timeout,
                ),
            )
        return self._async_client

    def chat(self, messages: Any, model: Optional[str] = None, **kwargs) -> str:
        """Send messages to Ollama chat endpoint."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        client = self._ensure_sync_client()
        model = model or self.config.default_model
        
        payload = {"model": model, "messages": messages, "stream": False, **kwargs}
        self.logger.debug("ollama chat payload", extra={"model": model, "msg_count": len(messages)})
        
        resp = client.post("/api/chat", json=payload, timeout=self.config.chat_timeout)
        resp.raise_for_status()
        result = resp.json()
        return result.get("message", {}).get("content", "")

    async def _async_chat(self, messages: List[Dict[str, Any]], model: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        client = await self._ensure_client()
        model = model or self.config.default_model

        payload = {"model": model, "messages": messages, "stream": False, **kwargs}
        self.logger.debug("ollama chat payload", extra={"model": model, "msg_count": len(messages)})

        resp = await client.post("/api/chat", json=payload, timeout=self.config.chat_timeout)
        resp.raise_for_status()
        return resp.json()

    async def embeddings(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        client = await self._ensure_client()
        model = model or self.config.default_model
        payload = {"model": model, "prompt": prompt}
        resp = await client.post("/api/embeddings", json=payload, timeout=self.config.embeddings_timeout)
        resp.raise_for_status()
        return resp.json()

    def health_check(self) -> bool:
        try:
            client = self._ensure_sync_client()
            resp = client.get("/api/tags", timeout=self.config.connection_timeout)
            return resp.status_code == 200
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def _check_ollama_health(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.config.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> List[str]:
        client = await self._ensure_client()
        resp = await client.get("/api/tags", timeout=self.config.connection_timeout)
        resp.raise_for_status()
        data = resp.json()
        return [m.get("name") for m in data.get("models", []) if m.get("name")]
    
    def get_available_models(self) -> List[str]:
        client = self._ensure_sync_client()
        resp = client.get("/api/tags", timeout=self.config.connection_timeout)
        resp.raise_for_status()
        data = resp.json()
        return [m.get("name") for m in data.get("models", []) if m.get("name")]