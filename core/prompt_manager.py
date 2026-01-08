"""Prompt Manager for editable system prompts."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages system prompt with override capability."""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.custom_prompt_path = self.data_dir / "prompts" / "custom_system.md"
        self.default_prompt_path = Path("core/prompts/llm.md")
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        prompt_dir = self.data_dir / "prompts"
        if not prompt_dir.exists():
            prompt_dir.mkdir(parents=True, exist_ok=True)
    
    def get_system_prompt(self) -> str:
        """Get active system prompt (custom override or default)."""
        if self.custom_prompt_path.exists():
            try:
                return self.custom_prompt_path.read_text(encoding="utf-8").strip()
            except Exception as e:
                logger.warning(f"Failed to read custom prompt: {e}")
        
        if self.default_prompt_path.exists():
            try:
                return self.default_prompt_path.read_text(encoding="utf-8").strip()
            except Exception as e:
                logger.warning(f"Failed to read default prompt: {e}")
        
        return "You are a helpful AI assistant."
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set custom system prompt (persisted to data/prompts/)."""
        self._ensure_dirs()
        self.custom_prompt_path.write_text(prompt, encoding="utf-8")
        logger.info("Custom system prompt saved")
    
    def reset_system_prompt(self) -> None:
        """Remove custom prompt, reverting to default."""
        if self.custom_prompt_path.exists():
            self.custom_prompt_path.unlink()
            logger.info("Custom system prompt removed, using default")
    
    def has_custom_prompt(self) -> bool:
        """Check if a custom prompt is set."""
        return self.custom_prompt_path.exists()


_prompt_manager: Optional[PromptManager] = None

def get_prompt_manager() -> PromptManager:
    """Get the global PromptManager instance."""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager
