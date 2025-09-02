# Development task automation
.PHONY: help install test lint format clean dev

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all packages in development mode
	pip install -e ./packages/core
	pip install -e ./packages/rag
	pip install -e ./packages/llms
	pip install -e ./packages/agents
	pip install -e ./packages/connectors
	pip install -e ./apps/assistant-api

test: ## Run tests
	cd apps/assistant-api && python -m pytest tests/

lint: ## Run linting
	cd apps/assistant-api && python -m ruff check .
	cd apps/assistant-api && python -m mypy app/

format: ## Format code
	cd apps/assistant-api && python -m black .
	cd apps/assistant-api && python -m ruff --fix .

clean: ## Clean build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

dev: install ## Start development server
	cd apps/assistant-api && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
