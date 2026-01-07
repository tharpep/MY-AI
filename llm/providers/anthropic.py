"""Anthropic Claude API Client"""

import json
import os
import urllib.request
import urllib.error
from typing import Optional, List, Any
from ..base_client import BaseLLMClient

def load_env_file():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env_file()


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Anthropic Claude client."""
        from core.config import get_config
        config = get_config()
        
        # Try multiple environment variable names for flexibility
        self.api_key = (
            api_key or 
            config.anthropic_api_key or 
            os.getenv('CLAUDE') or 
            os.getenv('ANTHROPIC_API_KEY')
        )
        
        if not self.api_key:
            raise ValueError("API key is required. Provide it directly or set CLAUDE environment variable.")
        
        self.base_url = "https://api.anthropic.com/v1/messages"
        # Get default model from config
        self.default_model = config.model_anthropic
        self.api_version = "2023-06-01"
    
    def chat(self, messages: Any, model: Optional[str] = None, **kwargs) -> str:
        """Send a message and get a response."""
        # Handle both string and list formats
        if isinstance(messages, str):
            # Convert single string to messages format
            messages_list = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            # Ensure all messages have the right format
            messages_list = []
            for msg in messages:
                if isinstance(msg, dict):
                    # Anthropic uses 'user' and 'assistant' roles
                    role = msg.get("role", "user")
                    if role == "system":
                        # Anthropic handles system messages differently
                        continue  # Skip system messages for now
                    elif role not in ["user", "assistant"]:
                        role = "user"
                    messages_list.append({
                        "role": role,
                        "content": msg.get("content", "")
                    })
                else:
                    messages_list.append({"role": "user", "content": str(msg)})
        else:
            messages_list = [{"role": "user", "content": str(messages)}]
        
        # Extract system message if present (Anthropic handles it separately)
        system_messages = [msg for msg in messages_list if msg.get("role") == "system"]
        system_content = system_messages[0]["content"] if system_messages else None
        
        # Filter out system messages from conversation (Anthropic handles them separately)
        conversation_messages = [msg for msg in messages_list if msg.get("role") != "system"]
        
        if not conversation_messages:
            raise ValueError("No conversation messages found")
        
        model = model or self.default_model
        
        # Prepare request with full conversation history
        data = {
            "model": model,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "messages": conversation_messages
        }
        
        # Add system message if provided
        if system_content:
            data["system"] = system_content
        
        # Make API request
        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(data).encode('utf-8'),
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": self.api_version
            }
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                # Extract text from response
                if "content" in result and len(result["content"]) > 0:
                    return result["content"][0].get("text", "")
                else:
                    raise Exception("Unexpected response format from Anthropic API")
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            try:
                error_data = json.loads(error_body)
                error_msg = error_data.get("error", {}).get("message", error_body)
            except:
                error_msg = error_body
            raise Exception(f"Anthropic API error: {error_msg}")
        except Exception as e:
            raise Exception(f"Error calling Anthropic API: {str(e)}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available Anthropic models."""
        return [
            "claude-opus-4-1-20250805",  # Latest Opus 4.1 (Aug 2025) - Most capable
            "claude-sonnet-4-5-20250929",  # Latest Sonnet 4.5 (Sep 2025) - Balanced performance
            "claude-haiku-4-5-20251001",  # Latest Haiku 4.5 (Oct 2025) - Fast, cost-effective
        ]

