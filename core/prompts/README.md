# LLM Prompts

Simple markdown files containing prompt templates. All prompts are passed through the gateway to providers.

## Architecture

**Simple Flow:**
1. Read prompt from `.md` file
2. Format with variables
3. Pass to `gateway.chat(prompt, ...)`
4. Gateway passes through to provider (all providers support string prompts)

## Usage

```python
from core.prompts import get_prompt, format_prompt
from llm.gateway import AIGateway

gateway = AIGateway()

# RAG: Read prompt, format, pass to gateway
rag_template = get_prompt("rag")
prompt = format_prompt(rag_template, context="...", question="What is Docker?")
answer = gateway.chat(prompt, provider=None, model=None)

# LLM: Read prompt, format, pass to gateway
llm_template = get_prompt("llm")
# Use as system message or prepend to user message
prompt = f"{llm_template}\n\nUser: Hello!"
response = gateway.chat(prompt, provider="ollama")
```

## Files

- `rag.md` - RAG prompt template (uses `{context}` and `{question}`)
- `llm.md` - General LLM system message/prompt

## Adding New Prompts

1. Create a new `.md` file in this directory
2. Use `{variable}` syntax for placeholders
3. Read with `get_prompt("filename")` (without .md extension)
4. Format and pass directly to `gateway.chat()`
