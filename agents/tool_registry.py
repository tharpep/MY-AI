"""Tool registry for managing available tools"""

from typing import Dict, Optional, List, Any
from .base_tool import BaseTool, ToolResult, ToolSchema
import logging

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing and executing tools."""
    
    def __init__(self):
        """Initialize empty tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._allowlist: Optional[List[str]] = None
    
    def register(self, tool: BaseTool) -> None:
        """Register a tool in the registry."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(tool_name)
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools (respecting allowlist)."""
        all_tools = list(self._tools.keys())
        if self._allowlist is None:
            return all_tools
        return [name for name in all_tools if name in self._allowlist]
    
    def set_allowlist(self, tool_names: Optional[List[str]]) -> None:
        """Set tool allowlist."""
        if tool_names is not None:
            for tool_name in tool_names:
                if tool_name not in self._tools:
                    logger.warning(f"Tool '{tool_name}' in allowlist is not registered")
        
        self._allowlist = tool_names
        logger.info(f"Tool allowlist set: {tool_names}")
    
    def is_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed (in allowlist)."""
        if tool_name not in self._tools:
            return False
        if self._allowlist is None:
            return True
        return tool_name in self._allowlist
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        validate: bool = True
    ) -> ToolResult:
        """Execute a tool with given parameters."""
        tool = self.get_tool(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found"
            )
        
        if not self.is_allowed(tool_name):
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' is not in allowlist"
            )
        
        if validate:
            is_valid, error_msg = tool.validate_parameters(parameters)
            if not is_valid:
                return ToolResult(
                    success=False,
                    error=f"Parameter validation failed: {error_msg}"
                )
        
        try:
            import time
            start_time = time.time()
            result = await tool.execute(**parameters)
            execution_time = (time.time() - start_time) * 1000
            
            if result.execution_time_ms is None:
                result.execution_time_ms = execution_time
            
            logger.info(f"Tool '{tool_name}' executed in {execution_time:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution failed: {e}", exc_info=True)
            return ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}"
            )
    


# Global tool registry instance
_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def reset_registry() -> None:
    """Reset the global tool registry (mainly for testing)."""
    global _registry
    _registry = None

