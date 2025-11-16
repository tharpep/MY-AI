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
### Phase 0 (MVP)
- 📚 **RAG Answer**: Search and cite from your personal document corpus
- 🔍 **Drive Search**: Find files across connected storage
- 🌐 **Web Search**: Fetch public references with attribution

### Phase 1 (Planned)
- 📅 **Calendar Integration**: Read events and scheduling data
- 🎵 **Spotify Lookup**: Access playlists and music data

### Phase 2 (Future)
- 💰 **Banking Read-Only**: Financial insights (optional, secure)

## API Endpoints
- `POST /v1/query` - Main query interface
- `POST /v1/ingest` - Add documents to knowledge base
- `POST /v1/chat` - Direct LLM gateway (local models)
- `POST /v1/embeddings` - Vectorize text for RAG

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
poetry install
```

### 4. Initial Setup (One-time)
```bash
# Activate Poetry shell (adds myai command to PATH)
poetry shell

# Setup virtual environment and install dependencies
myai setup
```

### 5. Use the CLI
After setup, use commands directly:

```bash
# Make sure you're in Poetry shell
poetry shell

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

### 6. Run the API Server (Optional)
```bash
# Development mode
poetry dev
# Or: make dev

# Or with Docker
make docker-dev
```

The API will be available at `http://localhost:8000`

## Security & Privacy
- ✅ Local processing by default
- ✅ No secrets in logs
- ✅ Parameterized queries only
- ✅ User data stays on your machine
- ✅ Optional cloud features require explicit consent

## Development Roadmap

### ✅ Step 0: Planning & Contracts (Complete)
- Software contracts and guardrails defined
- API surfaces and tool allowlists established
- Tier-2 routing strategy documented

### ✅ Step 1: API Scaffolding (Complete)
- Repository structure: Single-package format with `app/`, `llm/`, `rag/`, `agents/`, `connectors/`, `core/`
- Basic `/v1/query` endpoint with placeholder responses
- Health check endpoint
- LLM Gateway with Ollama integration

### ✅ Step 2: Local LLM Gateway (Complete)
- Ollama integration for local models
- `/v1/chat` and `/v1/embeddings` endpoints
- Configurable local/cloud model switching

### ✅ Step 3: RAG MVP (Complete)
- Document ingestion and chunking pipeline
- Qdrant vector storage with persistent collections
- Cited answer retrieval from personal corpus
- Interactive demos and CLI tools

### 📋 Step 4: AI Router (Planned)
- Natural language to `{tool, args}` JSON routing
- Schema validation and clarification flows
- Bounded LLM call counts (1-2 per request)

### 📋 Step 5: Tool Integration (Planned)
- Drive search, web search, calendar lookup
- Read-only tool implementations
- Tool execution transcripts and logging

### 📋 Step 6: Memory System (Planned)
- Short-term session memory
- Local storage with purge capabilities
- Foundation for future long-term memory

### 📋 Step 7: Dynamic Corpus (Future)
- AI-proposed document updates
- Human approval workflow
- Provenance tracking and diff management

## Project Structure
```
MY-AI/
├── app/                    # FastAPI application
│   ├── main.py            # Application entry point
│   ├── routes/            # API endpoints
│   └── services/          # Business logic
├── llm/                   # LLM functionality
│   ├── gateway.py         # LLM gateway
│   ├── local.py           # Ollama client
│   └── external.py        # External API adapters
├── rag/                   # Retrieval augmented generation
├── agents/                # AI routing and tool orchestration
├── connectors/            # External service integrations
├── core/                  # Shared utilities and schemas
├── tests/                 # Test files
├── pyproject.toml         # Poetry configuration
├── Makefile              # Development commands
└── docker-compose.yml    # Local development with Ollama
```

## Development Status
✅ **Steps 0-3 Complete** - Core API, LLM Gateway, and RAG system fully implemented  
🚧 **Step 4 In Progress** - AI Router for natural language to tool routing  
📋 **Steps 5-7 Planned** - Tool Integration, Memory System, Dynamic Corpus

See `/Documentation/` for detailed specifications and development roadmap.

## Quick Demo
Try the system locally without setting up the full API:

```bash
# Activate Poetry shell first
poetry shell

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

# Setup environment
myai setup
```

### Tab Completion
Enable tab completion for faster CLI usage:

```bash
# Make sure you're in Poetry shell
poetry shell

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
