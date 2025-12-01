"""
Pytest configuration and fixtures for LLM tests, including Ollama Local setup
"""
import pytest
import subprocess
import os
import time
import requests
import shutil
from pathlib import Path


@pytest.fixture(scope="session")
def check_ollama_installed():
    """
    Check if Ollama is installed on the system.

    Installation instructions:
    ```
    curl -fsSL https://ollama.com/install.sh | sh
    ollama --version
    ```
    """
    ollama_path = shutil.which("ollama")
    if ollama_path is None:
        pytest.skip(
            "Ollama is not installed. Install with: "
            "curl -fsSL https://ollama.com/install.sh | sh"
        )

    # Verify ollama can run
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            pytest.skip(f"Ollama is installed but not working properly: {result.stderr}")
    except Exception as e:
        pytest.skip(f"Failed to verify Ollama installation: {e}")

    return ollama_path


@pytest.fixture(scope="session")
def ollama_model_name():
    """
    Get the Ollama model name from environment or use default.

    Set LOCAL_MODEL_NAME environment variable or use default: qwen3-coder:latest
    """
    return os.getenv("LOCAL_MODEL_NAME", "qwen3-coder:latest")


@pytest.fixture(scope="session")
def ollama_model_port():
    """
    Get the Ollama server port from environment or use default.

    Set LOCAL_MODEL_PORT environment variable or use default: 11434
    """
    return os.getenv("LOCAL_MODEL_PORT", "11434")


@pytest.fixture(scope="session")
def ensure_ollama_model_pulled(check_ollama_installed, ollama_model_name):
    """
    Ensure the required Ollama model is pulled/downloaded.

    This will check if the model exists, and if not, attempt to pull it.
    Pulling a model can take several minutes depending on the model size.
    """
    # Check if model is already pulled
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if ollama_model_name in result.stdout:
            return True

        # Model not found, attempt to pull it
        print(f"\nPulling Ollama model: {ollama_model_name}")
        print("This may take several minutes...")

        pull_result = subprocess.run(
            ["ollama", "pull", ollama_model_name],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes timeout for pulling
        )

        if pull_result.returncode != 0:
            pytest.skip(
                f"Failed to pull Ollama model {ollama_model_name}: {pull_result.stderr}"
            )

        print(f"Successfully pulled model: {ollama_model_name}")
        return True

    except subprocess.TimeoutExpired:
        pytest.skip(f"Timeout while pulling Ollama model {ollama_model_name}")
    except Exception as e:
        pytest.skip(f"Error checking/pulling Ollama model: {e}")


@pytest.fixture(scope="session")
def ollama_server(check_ollama_installed, ensure_ollama_model_pulled, ollama_model_port):
    """
    Start Ollama server if not already running.

    This fixture ensures the Ollama server is running on the specified port.
    It will attempt to start the server if it's not running, and will stop it
    after tests complete if it was started by this fixture.
    """
    server_url = f"http://localhost:{ollama_model_port}"

    # Check if server is already running
    server_already_running = False
    try:
        response = requests.get(f"{server_url}/api/tags", timeout=2)
        if response.status_code == 200:
            server_already_running = True
            print(f"\nOllama server already running on port {ollama_model_port}")
    except requests.exceptions.RequestException:
        pass

    process = None

    if not server_already_running:
        # Start ollama server
        print(f"\nStarting Ollama server on port {ollama_model_port}...")

        try:
            # Start ollama serve in background
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "OLLAMA_HOST": f"127.0.0.1:{ollama_model_port}"}
            )

            # Wait for server to be ready (up to 30 seconds)
            for i in range(30):
                try:
                    response = requests.get(f"{server_url}/api/tags", timeout=2)
                    if response.status_code == 200:
                        print(f"Ollama server started successfully on port {ollama_model_port}")
                        break
                except requests.exceptions.RequestException:
                    time.sleep(1)
            else:
                if process:
                    process.terminate()
                pytest.skip(f"Failed to start Ollama server on port {ollama_model_port}")

        except Exception as e:
            if process:
                process.terminate()
            pytest.skip(f"Error starting Ollama server: {e}")

    yield server_url

    # Cleanup: stop server if we started it
    if process and not server_already_running:
        print("\nStopping Ollama server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


@pytest.fixture(scope="session")
def ollama_env_config(ollama_model_name, ollama_model_port):
    """
    Provide environment configuration for Ollama Local tests.

    This fixture returns a dictionary with the necessary environment variables
    that should be set for OllamaLocal to work properly.
    """
    return {
        "LOCAL_MODEL_NAME": ollama_model_name,
        "LOCAL_MODEL_PORT": ollama_model_port
    }


@pytest.fixture
def ollama_local_ready(ollama_server, ollama_env_config, monkeypatch):
    """
    Complete setup fixture that ensures Ollama is ready for testing.

    This fixture:
    1. Checks Ollama is installed
    2. Ensures the model is pulled
    3. Starts the server if needed
    4. Sets environment variables

    Use this fixture in tests that need OllamaLocal functionality.

    Example:
    ```python
    def test_ollama_local_ask(ollama_local_ready):
        from microbots.llm.ollama_local import OllamaLocal

        llm = OllamaLocal(
            system_prompt="You are a helpful assistant",
            model_name=ollama_local_ready["model_name"],
            model_port=ollama_local_ready["model_port"]
        )

        response = llm.ask("Say hello")
        assert response is not None
    ```
    """
    # Set environment variables
    for key, value in ollama_env_config.items():
        monkeypatch.setenv(key, value)

    # Return configuration for test use
    return {
        "server_url": ollama_server,
        "model_name": ollama_env_config["LOCAL_MODEL_NAME"],
        "model_port": ollama_env_config["LOCAL_MODEL_PORT"]
    }


@pytest.fixture
def mock_ollama_response():
    """
    Provide a mock Ollama server response for unit tests.

    This fixture is useful for unit tests that don't require an actual
    Ollama server running.

    Example:
    ```python
    def test_ollama_response_parsing(mock_ollama_response):
        # Use mock_ollama_response in your test
        pass
    ```
    """
    return {
        "model": "qwen3-coder:latest",
        "created_at": "2025-12-01T00:00:00.000000000Z",
        "response": '{"task_done": false, "command": "echo \'hello\'", "thoughts": "Executing echo"}',
        "done": True,
        "context": [],
        "total_duration": 1000000000,
        "load_duration": 500000000,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 200000000,
        "eval_count": 20,
        "eval_duration": 300000000
    }


# Marker for tests that require Ollama Local
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "ollama_local: mark test as requiring Ollama Local setup (deselect with '-m \"not ollama_local\"')"
    )
