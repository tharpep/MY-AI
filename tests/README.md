# Personal AI Assistant - Test Suite

This directory contains comprehensive tests for the Personal AI Assistant project.

## Test Structure

```
tests/
├── tests_api/           # FastAPI endpoint tests
├── tests_ai_providers/  # LLM provider tests (Ollama, Purdue)
├── tests_rag/          # RAG system tests
├── tests_tuning/       # Model tuning tests
├── run_tests.py        # Custom test runner
└── README.md          # This file
```

## Running Tests

### Using Makefile (Recommended)

```bash
# Run all tests
make test

# Run specific test categories
make test-api      # API endpoint tests
make test-ai       # AI provider tests  
make test-rag      # RAG system tests
make test-tuning   # Tuning tests

# Run with custom test runner (categorized output)
make test-quick
```

### Using pytest directly

```bash
# Run all tests
poetry run pytest tests/ -v

# Run specific categories
poetry run pytest tests/tests_api/ -v
poetry run pytest tests/tests_ai_providers/ -v
poetry run pytest tests/tests_rag/ -v
poetry run pytest tests/tests_tuning/ -v

# Run specific test file
poetry run pytest tests/tests_api/test_api.py -v

# Run with coverage
poetry run pytest tests/ --cov=app --cov=llm --cov=rag --cov=tuning
```

### Using custom test runner

```bash
# Run all tests with categorized output
python tests/run_tests.py

# Run specific test pattern
python tests/run_tests.py "test_api"
python tests/run_tests.py "test_gateway"
```

## Test Categories

### 1. API Tests (`tests_api/`)

Tests for FastAPI endpoints:

- **Health endpoints**: `/health/`, `/health/detailed`
- **LLM endpoints**: `/v1/models`, `/v1/chat/completions`, `/v1/embeddings`
- **RAG endpoints**: `/v1/query`, `/v1/ingest`, `/v1/stats`
- **Error handling**: Invalid JSON, missing fields, nonexistent endpoints
- **Integration tests**: Complete workflows

**Key Features:**
- Uses `AsyncClient` for async endpoint testing
- Mocks external dependencies (Ollama, Qdrant)
- Tests both success and error scenarios
- Validates response structure and status codes

### 2. AI Provider Tests (`tests_ai_providers/`)

Tests for LLM provider implementations:

#### Gateway Tests (`test_gateway.py`)
- Gateway initialization with different configs
- Provider selection logic
- Chat routing to correct providers
- Error handling for unavailable providers

#### Ollama Client Tests (`test_local.py`)
- Configuration management
- Health checks
- Chat functionality (sync/async)
- Embeddings generation
- Model listing
- Connection error handling

#### Purdue API Tests (`test_purdue_api.py`)
- API key management
- Chat requests (string/list messages)
- Model selection
- Error handling (API errors, HTTP errors)
- Available models listing

**Key Features:**
- Comprehensive mocking of HTTP clients
- Tests both sync and async operations
- Validates configuration handling
- Tests error scenarios

### 3. RAG Tests (`tests_rag/`)

Tests for Retrieval Augmented Generation system:

- **RAG initialization**: Collection setup, configuration
- **Document ingestion**: File processing, chunking, indexing
- **Vector search**: Similarity search, ranking
- **Query processing**: Context retrieval, answer generation
- **Statistics**: Collection metrics, document counts

**Key Features:**
- Uses real document files for testing
- Tests both persistent and in-memory storage
- Validates search result quality
- Tests complete RAG workflows

### 4. Tuning Tests (`tests_tuning/`)

Tests for model fine-tuning functionality:

#### Basic Tuning Tests (`test_basic_tuning.py`)
- Tuner initialization and configuration
- Model loading and mapping
- Data preparation and tokenization
- Trainer setup with optimized parameters
- Error handling for missing components

#### Demo Tests (`test_demo.py`)
- Demo workflow execution
- Training data selection
- Configuration integration
- Error handling during training

**Key Features:**
- Mocks heavy ML dependencies (transformers, datasets)
- Tests hardware optimization (laptop vs PC settings)
- Validates training parameter configuration
- Tests complete tuning workflows

## Test Configuration

### Prerequisites

1. **Dependencies**: All test dependencies are included in `pyproject.toml`
2. **Environment**: Tests use mocked external services (Ollama, Qdrant)
3. **Data**: RAG tests require sample documents in `data/documents/`

### Test Data

- **RAG Tests**: Use markdown files in `data/documents/` directory
- **Tuning Tests**: Use predefined sample training texts
- **API Tests**: Use mocked responses for external services

### Mocking Strategy

- **External APIs**: Ollama, Purdue GenAI Studio
- **Vector Database**: Qdrant operations
- **ML Libraries**: Transformers, datasets (for tuning tests)
- **File System**: Document reading operations

## Writing New Tests

### Test Structure

```python
import pytest
from unittest.mock import patch, MagicMock

class TestFeature:
    """Test class for specific feature"""
    
    @pytest.fixture
    def setup_data(self):
        """Setup test data"""
        return {"test": "data"}
    
    def test_basic_functionality(self, setup_data):
        """Test basic functionality"""
        # Arrange
        # Act  
        # Assert
        assert True
    
    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test async functionality"""
        # Test async code
        assert True
```

### Best Practices

1. **Use descriptive test names**: `test_should_return_error_when_invalid_input`
2. **Mock external dependencies**: Don't rely on external services
3. **Test both success and failure cases**: Cover error scenarios
4. **Use fixtures for setup**: Reusable test data and configuration
5. **Group related tests**: Use test classes for organization
6. **Validate response structure**: Check all required fields
7. **Use async/await properly**: For async endpoint tests

### Adding New Test Categories

1. Create new directory: `tests/tests_new_feature/`
2. Add `__init__.py` file
3. Create test files: `test_feature.py`
4. Update `run_tests.py` to include new category
5. Add Makefile target: `test-new-feature`

## Continuous Integration

Tests are designed to run in CI environments:

- **No external dependencies**: All external services are mocked
- **Deterministic**: Tests produce consistent results
- **Fast execution**: Optimized for quick feedback
- **Comprehensive coverage**: All major functionality tested

## Troubleshooting

### Common Issues

1. **Import errors**: Check that all imports use correct paths (`llm.` not `ai_providers.`)
2. **Mock failures**: Ensure mocks match actual method signatures
3. **Async test issues**: Use `@pytest.mark.asyncio` for async tests
4. **Configuration errors**: Check that config imports use `core.config`

### Debug Mode

Run tests with verbose output and no capture:

```bash
poetry run pytest tests/ -v -s --tb=long
```

### Test Specific Functionality

```bash
# Test specific method
poetry run pytest tests/ -k "test_chat_completions" -v

# Test specific class
poetry run pytest tests/ -k "TestAIGateway" -v

# Test with pattern
poetry run pytest tests/ -k "gateway" -v
```

## Coverage

To generate test coverage reports:

```bash
poetry run pytest tests/ --cov=app --cov=llm --cov=rag --cov=tuning --cov-report=html
```

This generates an HTML coverage report in `htmlcov/` directory.
