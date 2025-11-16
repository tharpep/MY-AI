"""
Simple AI Gateway
Routes requests to AI providers (Purdue GenAI Studio, Local Ollama)
Designed to be easily extended for additional providers
"""

import os
import asyncio
from typing import Dict, Any, Optional, List
from .purdue_api import PurdueGenAI
from .local import OllamaClient, OllamaConfig
from core.config import get_config

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file"""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env_file()


class AIGateway:
    """Simple gateway for AI requests"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize gateway with configuration
        
        Args:
            config: Dictionary with provider configurations
                   If None, will try to load from environment variables and config.py
        """
        self.providers = {}
        self.config = get_config()
        self._setup_providers(config or {})
    
    def _setup_providers(self, config: Dict[str, Any]):
        """Setup available AI providers"""
        # Setup Purdue provider
        if "purdue" in config:
            api_key = config["purdue"].get("api_key")
            self.providers["purdue"] = PurdueGenAI(api_key)
        elif self.config.purdue_api_key:
            self.providers["purdue"] = PurdueGenAI()
        elif os.getenv('PURDUE_API_KEY'):
            self.providers["purdue"] = PurdueGenAI()
        
        # Setup Local Ollama provider
        if "ollama" in config:
            ollama_config = OllamaConfig(
                base_url=config["ollama"].get("base_url", self.config.ollama_base_url),
                default_model=config["ollama"].get("default_model", self.config.model_name)
            )
            self.providers["ollama"] = OllamaClient(ollama_config)
        elif self.config.provider_type == "local" and self.config.provider_name == "ollama":
            ollama_config = OllamaConfig(
                base_url=self.config.ollama_base_url,
                default_model=self.config.model_name
            )
            self.providers["ollama"] = OllamaClient(ollama_config)
    
    def chat(self, message: str, provider: Optional[str] = None, model: Optional[str] = None) -> str:
        """
        Send a chat message to specified AI provider
        
        Args:
            message: Your message to the AI
            provider: AI provider to use (auto-selects based on availability)
            model: Model to use (uses provider default if not specified)
            
        Returns:
            str: AI response
        """
        # Auto-select provider based on config
        if provider is None:
            provider = self.config.provider_name
            # Fallback logic if primary provider not available
            if provider not in self.providers:
                if self.config.provider_fallback and self.config.provider_fallback in self.providers:
                    provider = self.config.provider_fallback
                elif "ollama" in self.providers:
                    provider = "ollama"
                elif "purdue" in self.providers:
                    provider = "purdue"
                else:
                    raise Exception(f"No providers available. Configure provider in config or set API keys in .env")
        
        if provider not in self.providers:
            available = ", ".join(self.providers.keys())
            raise Exception(f"Provider '{provider}' not available. Available: {available}")
        
        provider_client = self.providers[provider]
        
        # Handle different provider types
        if provider == "ollama":
            return self._chat_ollama(provider_client, message, model)
        else:
            # Use config model if no model specified
            model = model or self.config.model_name
            return provider_client.chat(message, model)
    
    def _chat_ollama(self, client: OllamaClient, message: str, model: Optional[str] = None) -> str:
        """Helper to handle Ollama calls"""
        # Now that OllamaClient.chat() is synchronous, we can call it directly
        return client.chat(message, model=model)
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.providers.keys())
    
    async def embeddings(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate embeddings for text
        
        Args:
            prompt: Text to embed
            model: Model to use (optional)
            
        Returns:
            Dictionary with embedding data
        """
        # Use Ollama for embeddings if available
        if "ollama" in self.providers:
            ollama_client = self.providers["ollama"]
            return await ollama_client.embeddings(prompt, model)
        else:
            raise Exception("No embedding provider available")


if __name__ == "__main__":
    try:
        gateway = AIGateway()
        response = gateway.chat("Hello! What is your name?")
        print(f"AI Response: {response}")
        print(f"Available providers: {gateway.get_available_providers()}")
    except Exception as e:
        print(f"Error: {e}")
