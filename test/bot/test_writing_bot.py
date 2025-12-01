"""
This test uses the WritingBot to solve https://github.com/SWE-agent/test-repo/issues/1
The issue is a simple syntax correction issue from original SWE-bench's test-repo.

This test can run with either Azure OpenAI or Ollama Local (qwen3-coder:latest).

Usage:
------
# Run only Azure OpenAI test (skips Ollama):
pytest test/bot/test_writing_bot.py::test_writing_bot_azure -v

# Run only Ollama Local test (requires Ollama installed with qwen3-coder:latest):
pytest test/bot/test_writing_bot.py -v -m ollama_local

# Run all tests except Ollama:
pytest test/bot/test_writing_bot.py -v -m "not ollama_local"

# Run all integration tests including both Azure and Ollama:
pytest test/bot/test_writing_bot.py -v
"""

import os
import sys

import pytest
# Add src directory to path to import from local source
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

import logging
logging.basicConfig(level=logging.INFO)

from microbots import WritingBot, BotRunResult

@pytest.mark.integration
def test_writing_bot_azure(test_repo, issue_1):
    """Test WritingBot with Azure OpenAI model"""
    issue_text = issue_1[0]
    verify_function = issue_1[1]

    writingBot = WritingBot(
        model="azure-openai/mini-swe-agent-gpt5",
        folder_to_mount=str(test_repo)
    )

    response: BotRunResult = writingBot.run(
        issue_text, timeout_in_seconds=300
    )

    print(f"Status: {response.status}, Result: {response.result}, Error: {response.error}")

    verify_function(test_repo)


@pytest.mark.integration
@pytest.mark.ollama_local
def test_writing_bot_ollama(test_repo, issue_1, ollama_local_ready):
    """Test WritingBot with Ollama Local qwen3-coder:latest model"""
    issue_text = issue_1[0]
    verify_function = issue_1[1]

    # Get the model name and port from the fixture
    model_name = ollama_local_ready["model_name"]
    model_port = ollama_local_ready["model_port"]

    os.environ["LOCAL_MODEL_NAME"] = model_name
    os.environ["LOCAL_MODEL_PORT"] = str(model_port)

    writingBot = WritingBot(
        model=f"ollama-local/{model_name}",
        folder_to_mount=str(test_repo)
    )

    response: BotRunResult = writingBot.run(
        issue_text, timeout_in_seconds=300
    )

    print(f"Status: {response.status}, Result: {response.result}, Error: {response.error}")

    verify_function(test_repo)