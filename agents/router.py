"""Tier-2 router for tool selection"""

from typing import Dict, Any, Optional
from .tool_registry import get_registry
import logging

logger = logging.getLogger(__name__)


class ToolRouter:
    """Router that analyzes user intent and selects appropriate tools."""
    
    def __init__(self, gateway=None):
        """Initialize tool router."""
        self.gateway = gateway
        self.registry = get_registry()
    
    def _get_gateway(self):
        if self.gateway is None:
            from llm.gateway import AIGateway
            self.gateway = AIGateway()
        return self.gateway
    
    async def route(
        self,
        message: str,
        available_tools: Optional[list[str]] = None
    ) -> Dict[str, Any]:
        """Route user message to appropriate tool or direct response."""
        # TODO: Implement LLM-based intent analysis in future phase
        
        if available_tools is None:
            available_tools = self.registry.get_available_tools()
        
        question_words = ["what", "when", "where", "who", "why", "how", "which"]
        message_lower = message.lower()
        
        is_question = any(word in message_lower for word in question_words)
        
        if "rag_answer" in available_tools and is_question:
            return {
                "tool": "rag_answer",
                "parameters": {"query": message, "top_k": 5},
                "use_rag": True,
                "direct_response": False
            }
        
        return {
            "tool": None,
            "parameters": {},
            "use_rag": False,
            "direct_response": True
        }
    
    def validate_tool_plan(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Validate a tool execution plan."""
        tool = self.registry.get_tool(tool_name)
        if tool is None:
            return False, f"Tool '{tool_name}' not found"
        
        if not self.registry.is_allowed(tool_name):
            return False, f"Tool '{tool_name}' is not in allowlist"
        
        is_valid, error_msg = tool.validate_parameters(parameters)
        return is_valid, error_msg
