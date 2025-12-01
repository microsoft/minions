# Ollama Local Testing Setup

This directory contains pytest fixtures and tests for the OllamaLocal LLM implementation.

## Prerequisites

### 1. Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version
```

### 2. Pull a Model

```bash
# Pull the default model (codellama:latest)
ollama pull codellama:latest

# Or pull a different model
ollama pull llama2:latest
```

### 3. Set Environment Variables (Optional)

Create a `.env` file in the project root or set these environment variables:

```bash
# Optional: defaults are shown
LOCAL_MODEL_NAME=codellama:latest
LOCAL_MODEL_PORT=11434
```

## Running Tests

### Run all tests (including unit tests that don't require Ollama server)

```bash
pytest test/llm/test_ollama_local.py -v
```

### Run only integration tests (requires Ollama server)

```bash
pytest test/llm/test_ollama_local.py -v -m ollama_local
```

### Skip integration tests (run only unit tests)

```bash
pytest test/llm/test_ollama_local.py -v -m "not ollama_local"
```

## Fixtures Overview

The `conftest.py` file provides several fixtures to manage Ollama setup:

### Session-scoped Fixtures

- **`check_ollama_installed`**: Verifies Ollama is installed, skips tests if not
- **`ollama_model_name`**: Gets model name from environment or uses default
- **`ollama_model_port`**: Gets port from environment or uses default (11434)
- **`ensure_ollama_model_pulled`**: Ensures the model is downloaded (auto-pulls if needed)
- **`ollama_server`**: Starts Ollama server if not running, stops it after tests
- **`ollama_env_config`**: Provides environment configuration dictionary

### Function-scoped Fixtures

- **`ollama_local_ready`**: Complete setup fixture that:
  - Checks installation
  - Ensures model is pulled
  - Starts server
  - Sets environment variables
  - Returns config dict for tests

- **`mock_ollama_response`**: Mock response for unit tests without actual server

## Usage Examples

### Unit Test (No Server Required)

```python
@patch('microbots.llm.ollama_local.requests.post')
def test_my_feature(mock_post):
    """Test without actual Ollama server"""
    mock_post.return_value = Mock(
        status_code=200,
        json=lambda: {"response": '{"task_done": false, "command": "test", "result": null}'}
    )

    ollama = OllamaLocal(
        system_prompt="Test",
        model_name="codellama:latest",
        model_port="11434"
    )
    result = ollama.ask("test message")
    assert result is not None
```

### Integration Test (Requires Server)

```python
@pytest.mark.ollama_local
def test_with_real_server(ollama_local_ready):
    """Test with actual Ollama server"""
    ollama = OllamaLocal(
        system_prompt="You are a helpful assistant",
        model_name=ollama_local_ready["model_name"],
        model_port=ollama_local_ready["model_port"]
    )

    response = ollama.ask("Say hello")
    assert isinstance(response, LLMAskResponse)
```

## Troubleshooting

### Tests are skipped with "Ollama is not installed"

Install Ollama following the prerequisites above.

### Tests timeout during model pulling

The first time tests run, they may need to pull the model (several GB). This can take 5-10 minutes depending on your internet connection. Subsequent runs will be fast.

### Server port already in use

If you're running Ollama server manually, the fixture will detect it and use the existing server. Otherwise, set a different port:

```bash
export LOCAL_MODEL_PORT=11435
```

### Model not found

Ensure the model is pulled:

```bash
ollama pull codellama:latest
# or
ollama list  # to see available models
```

## Continuous Integration

For CI/CD pipelines, you may want to:

1. Pre-pull the model in a setup step
2. Start the Ollama server as a background service
3. Skip integration tests if Ollama is not available:

```yaml
# Example GitHub Actions
- name: Setup Ollama
  run: |
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull codellama:latest
    ollama serve &

- name: Run tests
  run: pytest test/llm/test_ollama_local.py -v
```

Or skip integration tests:

```yaml
- name: Run unit tests only
  run: pytest test/llm/test_ollama_local.py -v -m "not ollama_local"
```
