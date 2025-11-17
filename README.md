# My-AI
Personal AI Assistant - Local-First Intelligence Platform

## Overview
A privacy-focused personal AI assistant that runs locally first, with optional cloud fallbacks. Built around Retrieval Augmented Generation (RAG) with your personal documents and integrated tools for productivity.

## Architecture
- **Local-First**: Primary processing using local models (Ollama/vLLM)
- **Cloud Fallback**: Optional external APIs (GPT/Claude) when needed
- **Tool-Based**: Extensible system with read-only integrations
- **Privacy-Centric**: Your data stays local by default

## Core Features

### Current Features (Implemented)
- **RAG Answer Tool**: Search and cite from your personal document corpus
- **OpenAI-Compatible Chat**: `/v1/chat/completions` with provider/model selection
- **Request Logging**: SQLite database for request tracking and debugging
- **Tool Foundation**: Extensible tool system with registry and execution engine
- **Document Ingestion**: Add documents to RAG knowledge base
- **Health Monitoring**: Health check endpoints with detailed status

### Planned Features
- **Web Search Tool**: Fetch public references with attribution
- **Drive Search Tool**: Find files across connected storage
- **Calendar Integration**: Read events and scheduling data
- **Spotify Lookup**: Access playlists and music data (v2)
- **Banking Read-Only**: Financial insights (v2, optional, secure)

## API Endpoints

### OpenAI-Compatible Endpoints
- `POST /v1/chat/completions` - Chat completions with provider/model override support
- `POST /v1/embeddings` - Generate text embeddings
- `GET /v1/models` - List available models and providers

### RAG & Document Management
- `POST /v1/query` - RAG-powered question answering with citations
- `POST /v1/ingest` - Ingest documents into knowledge base
- `GET /v1/stats` - RAG system statistics

### Health & Monitoring
- `GET /health/` - Basic health check
- `GET /health/detailed` - Detailed health with component status

**Note:** All endpoints include request ID tracking for debugging and tracing.

## Quick Start

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running
- Poetry (recommended) or pip

### 1. Install Ollama
Download and install Ollama from [ollama.ai](https://ollama.ai/)

### 2. Pull Required Models

#### For Smaller PCs (Laptops/No Dedicated GPU)
```bash
# Default model
ollama pull qwen3:1.7b

# Fallback model
ollama pull llama3.2:1b

# Coding assistance
ollama pull qwen2.5-coder:1.5b-instruct

# Reasoning model
ollama pull deepseek-r1:1.5b
```

#### For Computers with Dedicated GPUs
```bash
# Primary model
ollama pull qwen3:8b

# Fallback model
ollama pull llama3.1:8b

# Coding assistance
ollama pull qwen2.5-coder:7b

# Reasoning model
ollama pull deepseek-r1:7b
```

### 3. Clone and Setup
```bash
git clone https://github.com/yourusername/MY-AI.git
cd MY-AI
```

### 4. Install Poetry and Setup
```bash
# Install Poetry
python -m pip install --user poetry

# Install Poetry shell plugin (required for Poetry 2.0+)
poetry self add poetry-plugin-shell

# Install project dependencies (creates venv and installs packages)
poetry install

# Activate Poetry shell (adds myai command to PATH)
poetry shell
```

### 5. Use the CLI
Commands are available after activating the Poetry shell:

```bash
# Interactive chat
myai chat

# Chat with specific provider/model
myai chat --provider ollama --model qwen3:8b

# Run demos
myai demo rag
myai demo llm
myai demo api

# Run tests
myai test --all
myai test tests_api

# Show configuration
myai config

# Get help
myai --help
myai chat --help
```

**Note:** If you prefer not to use the shell plugin, prefix commands with `poetry run` (e.g., `poetry run myai chat`).

### 6. Run the API Server (Optional)
```bash
# Development mode
poetry dev
# Or: make dev

# Or with Docker
make docker-dev
```

The API will be available at `http://localhost:8000`

### 7. API Usage Examples

#### Chat Completions (OpenAI-Compatible)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "provider": "ollama",
    "model": "llama3.2:1b"
  }'
```

#### RAG Query
```bash
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is Docker?",
    "context_limit": 5
  }'
```

#### Document Ingestion
```bash
curl -X POST "http://localhost:8000/v1/ingest?folder_path=./data/documents"
```

#### List Available Models
```bash
curl http://localhost:8000/v1/models
```

All responses include a `request_id` field for tracing. Check `./data/api_logs.db` for request history.

## CLI Commands

### Setup Commands
- `myai setup-poetry [--install]` - Check/install Poetry and shell plugin

### Usage Commands
- `myai chat` - Interactive chat with AI
- `myai demo [rag|llm|api|tuning]` - Run automated demos
- `myai test [--all] [category]` - Run tests
- `myai config` - Show current configuration

For help on any command: `myai <command> --help`

## Features

### API Features
- **OpenAI-Compatible**: Drop-in replacement for OpenAI API endpoints
- **Provider Override**: Specify which AI provider to use per request
- **Request Tracking**: All requests logged with unique IDs for debugging
- **Structured Errors**: Proper HTTP status codes and error messages
- **Token Estimation**: Rough token counting for usage tracking

### Tool System
- **Extensible Architecture**: Easy to add new tools by inheriting `BaseTool`
- **Tool Registry**: Centralized tool management with allowlist security
- **Parameter Validation**: Automatic validation before tool execution
- **Execution Tracking**: Performance metrics and error handling

### Request Logging
- **SQLite Database**: All API requests stored locally in `./data/api_logs.db`
- **Request IDs**: Every request gets a unique ID for tracing
- **Performance Metrics**: Response times tracked for all endpoints
- **Error Tracking**: Failed requests logged with error details

## Security & Privacy
- Local processing by default
- No secrets in logs (PII redaction)
- Parameterized queries only
- User data stays on your machine
- Optional cloud features require explicit consent
- Request logging stored locally (SQLite)
- Tool allowlist for security control

## Development Roadmap

### Step 0: Planning & Contracts (Complete)
- Software contracts and guardrails defined
- API surfaces and tool allowlists established
- Tier-2 routing strategy documented

### Step 1: API Scaffolding (Complete)
- Repository structure: Single-package format with `app/`, `llm/`, `rag/`, `agents/`, `connectors/`, `core/`
- Basic `/v1/query` endpoint with placeholder responses
- Health check endpoint
- LLM Gateway with Ollama integration

### Step 2: Local LLM Gateway (Complete)
- Ollama integration for local models
- OpenAI-compatible `/v1/chat/completions` and `/v1/embeddings` endpoints
- Configurable local/cloud model switching
- Provider override support in requests

### Step 3: RAG MVP (Complete)
- Document ingestion and chunking pipeline
- Qdrant vector storage with persistent collections
- Cited answer retrieval from personal corpus
- Interactive demos and CLI tools

### Step 4: API Enhancements (Complete)
- Route Organization: Separated endpoints into logical modules (llm, query, ingest, health)
- Error Handling: Comprehensive HTTP status codes and structured error responses
- Request Logging: SQLite database for request tracking and debugging
- Provider Selection: Chat endpoint supports provider/model override
- Request IDs: All requests tracked with unique IDs for tracing

### Step 5: Tool Foundation (Complete)
- Base Tool Interface: Abstract class for all tools
- Tool Registry: Centralized tool management with allowlist support
- Tool Execution Engine: Validation, execution, and error handling
- RAG Answer Tool: First tool implementation for RAG-powered queries
- Tool Router: Intent analysis and tool selection (basic heuristic routing)

### Step 6: Tool Integration (In Progress)
- Web search tool implementation
- Drive search tool implementation
- Calendar lookup tool implementation
- Tool execution transcripts and logging

### Step 7: Memory System (Planned)
- Short-term session memory
- Local storage with purge capabilities
- Foundation for future long-term memory

### Step 8: Dynamic Corpus (Future)
- AI-proposed document updates
- Human approval workflow
- Provenance tracking and diff management

## Project Structure
```
MY-AI/
├── app/                    # FastAPI application
│   ├── main.py            # Application entry point with lifecycle management
│   ├── db.py              # SQLite request logging database
│   └── routes/            # API endpoint modules
│       ├── health.py      # Health check endpoints
│       ├── llm.py         # OpenAI-compatible LLM endpoints
│       ├── query.py       # RAG query endpoints
│       └── ingest.py      # Document ingestion endpoints
├── llm/                   # LLM functionality
│   ├── gateway.py         # AI Gateway (routes to providers)
│   ├── local.py           # Ollama client
│   └── purdue_api.py      # Purdue GenAI Studio client
├── rag/                   # Retrieval augmented generation
│   ├── rag_setup.py       # RAG system orchestrator
│   ├── vector_store.py    # Qdrant vector database operations
│   └── document_ingester.py  # Document processing pipeline
├── agents/                # AI routing and tool orchestration
│   ├── base_tool.py       # Base tool interface
│   ├── tool_registry.py   # Tool management and execution
│   ├── router.py          # Intent analysis and routing
│   └── tools/             # Tool implementations
│       └── rag_answer.py  # RAG answer tool
├── connectors/            # External service integrations (stubs)
├── core/                  # Shared utilities and schemas
│   ├── config.py          # Unified configuration (Pydantic Settings)
│   └── schemas/           # Pydantic models
├── cli/                   # Command-line interface
│   ├── main.py            # CLI entry point (Typer)
│   └── commands/          # CLI command modules
├── tests/                 # Test files
├── data/                  # Data directory
│   ├── documents/         # Source documents for RAG
│   └── api_logs.db        # SQLite request logs (auto-created)
├── pyproject.toml         # Poetry configuration
└── docker-compose.yml     # Local development with Ollama
```

## Development Status
**Steps 0-5 Complete** - Core API, LLM Gateway, RAG system, API enhancements, and tool foundation fully implemented  
**Step 6 In Progress** - Tool implementations (web_search, drive_search, calendar_lookup)  
**Steps 7-8 Planned** - Memory System, Dynamic Corpus

### Recent Updates
- Request Logging: All API requests logged to SQLite for debugging and analytics
- Enhanced Error Handling: Proper HTTP status codes and structured error responses
- Provider Override: Chat endpoint supports explicit provider/model selection
- Tool System: Extensible tool foundation with registry and execution engine
- Route Organization: Clean separation of endpoints into logical modules

See `/Documentation/` for detailed specifications and development roadmap.

## Quick Demo
Try the system locally without setting up the full API:

```bash
# Interactive chat with AI
myai chat

# Chat with specific provider/model
myai chat --provider ollama --model qwen3:8b

# RAG demo (automated)
myai demo rag

# LLM demo (automated)
myai demo llm

# Run all tests
myai test --all

# Run specific test category
myai test tests_api

# Show configuration
myai config
```

### Tab Completion
Enable tab completion for faster CLI usage:

```bash
# For bash/zsh
myai --install-completion bash
# Then add to ~/.bashrc or ~/.zshrc:
# eval "$(_MYAI_COMPLETE=bash_source myai)"

# For PowerShell (Windows)
myai --install-completion powershell
# Then add to your PowerShell profile

# For fish
myai --install-completion fish | source
```

## Configuration

Configuration is managed through `core/config.py` and `.env` file:

```bash
# Example .env file
PROVIDER_TYPE=local
PROVIDER_NAME=ollama
MODEL_DEFAULT=llama3.2:1b
OLLAMA_BASE_URL=http://localhost:11434
PURDUE_API_STUDIO=your_purdue_key_here  # Optional
CLAUDE=your_claude_key_here  # Optional
```

View current configuration:
```bash
myai config
```

## Request Logging

All API requests are automatically logged to a SQLite database at `./data/api_logs.db`. This includes:
- Request ID, timestamp, endpoint, method
- Status code, response time
- Provider and model used (for LLM endpoints)
- Token usage (for chat endpoints)
- Error information (if any)

Query the logs programmatically:
```python
from app.db import get_request_by_id, get_recent_requests

# Get specific request
request = get_request_by_id("req_abc123")

# Get recent requests
recent = get_recent_requests(limit=100)
```

## Contributing
This is a personal project following a structured development path. See documentation for implementation details and architecture decisions.

### Development Commands
```bash
make install    # Install dependencies
make test       # Run all tests
make lint       # Check code quality
make format     # Format code
make dev        # Start development server
make docker-dev # Run with Docker
```

## License
This project is open source. Please check the LICENSE file for details.
