"""
LLM Prompts Module

Simple markdown prompt files that can be read and used.
"""

import os
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def get_prompt(name: str) -> str:
    """
    Read a prompt from a .md file.
    
    Args:
        name: Prompt name (without .md extension)
        
    Returns:
        Prompt content as string
    """
    prompt_path = _PROMPTS_DIR / f"{name}.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    return prompt_path.read_text(encoding="utf-8").strip()


def format_prompt(template: str, **kwargs) -> str:
    """
    Format a prompt template with variables.
    
    Args:
        template: Prompt template string with {variable} placeholders
        **kwargs: Variables to substitute
        
    Returns:
        Formatted prompt string
    """
    return template.format(**kwargs)
