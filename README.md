# My-AI: Local-First Intelligence Platform

A privacy-focused personal AI assistant built for local-first execution with optional cloud fallbacks. Designed around Retrieval Augmented Generation (RAG) to leverage your personal documents and integrated tools for enhanced productivity.

## Key Features

- **Privacy-Centric**: Data stays local by default. No external data sharing without explicit intent.
- **Local Intelligence**: Powered by Ollama/vLLM for primary processing, ensuring low latency and offline capability.
- **RAG Capabilities**: Search and cite from your personal document corpus with built-in document ingestion and vector storage.
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI endpoints (`/v1/chat/completions`), supporting any compatible client.
- **Extensible Tool System**: Modular architecture for integrating read-only tools and services.
- **Detailed Monitoring**: SQLite-backed request logging for debugging and performance tracking.

## Getting Started

### Prerequisites
- **Python 3.11+**
- **[Ollama](https://ollama.ai/)** (installed and running)
- **Poetry** (recommended) or pip

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/MY-AI.git
    cd MY-AI
    ```

2.  **Pull required models** (adjust based on your hardware):
    ```bash
    # Standard Setup (8GB+ RAM)
    ollama pull qwen3:1.7b
    ollama pull llama3.2:1b
    ```

3.  **Install dependencies**:
    ```bash
    poetry install
    ```

4.  **Initialize environment**:
    ```bash
    poetry shell
    myai setup
    ```

## Usage

### Command Line Interface (CLI)

The `myai` CLI provides direct access to chat, configuration, and demos.

```bash
# Interactive Chat
myai chat

# Chat with specific model
myai chat --provider ollama --model qwen3:8b

# Ingest documents for RAG
myai ingest --folder ./data/documents

# View Configuration
myai config
```

### API Server

Start the API server for external client integration:

```bash
# Start server
make dev
# API available at http://localhost:8000
```

**Common Endpoints:**
- `POST /v1/chat/completions`: Standard chat completion.
- `POST /v1/query`: RAG-specific endpoint with citations.
- `GET /v1/models`: List available models.
- `GET /health/detailed`: System health status.

## Configuration

Configuration is managed via `core/config.py` and environment variables (`.env`).

**Example `.env`**:
```env
PROVIDER_TYPE=local
PROVIDER_NAME=ollama
MODEL_DEFAULT=llama3.2:1b
OLLAMA_BASE_URL=http://localhost:11434
```

## Project Structure

```
MY-AI/
├── app/       # FastAPI application & routes
├── agents/    # Tool orchestration & routing
├── cli/       # Command-line interface
├── core/      # Config & shared utilities
├── llm/       # LLM gateway & providers
├── rag/       # RAG pipeline & vector store
└── tests/     # Test suite
```

## License

This project is open source. Please check the `LICENSE` file for details.
