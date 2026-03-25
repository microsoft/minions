"""
Unit tests for CopilotApi class
"""
import pytest
import json
import sys
import os
import asyncio
import importlib
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import asdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

# Mock the copilot module before importing CopilotApi, since github-copilot-sdk
# is an optional dependency that won't be installed in the test environment.
_mock_copilot = MagicMock()
_mock_copilot.PermissionHandler = MagicMock()
_mock_copilot.PermissionHandler.approve_all = MagicMock()
_mock_copilot.CopilotClient = MagicMock
_mock_copilot_types = MagicMock()
_mock_copilot_types.SubprocessConfig = MagicMock
sys.modules["copilot"] = _mock_copilot
sys.modules["copilot.types"] = _mock_copilot_types

# Now safe to import — the module-level `from copilot import ...` will resolve
# against our mock.
if "microbots.llm.copilot_api" in sys.modules:
    importlib.reload(sys.modules["microbots.llm.copilot_api"])
from microbots.llm.copilot_api import CopilotApi
from microbots.llm.llm import LLMAskResponse, LLMInterface, llm_output_format_str


@pytest.fixture
def mock_copilot_session():
    """Create a mock Copilot session with send_and_wait."""
    session = AsyncMock()
    session.disconnect = AsyncMock()

    # Default response from send_and_wait
    default_response = Mock()
    default_response.data.content = json.dumps({
        "task_done": False,
        "thoughts": "Thinking about the task",
        "command": "ls -la"
    })
    session.send_and_wait = AsyncMock(return_value=default_response)
    return session


@pytest.fixture
def mock_copilot_client(mock_copilot_session):
    """Create a mock CopilotClient."""
    client = AsyncMock()
    client.start = AsyncMock()
    client.stop = AsyncMock()
    client.create_session = AsyncMock(return_value=mock_copilot_session)
    return client


@pytest.fixture
def copilot_api(mock_copilot_client):
    """Create a CopilotApi instance with mocked client."""
    with patch("microbots.llm.copilot_api.CopilotClient", return_value=mock_copilot_client):
        api = CopilotApi(
            system_prompt="You are a test assistant",
            model_name="gpt-4.1",
        )
        yield api
        api.close()


@pytest.mark.unit
class TestCopilotApiInitialization:
    """Tests for CopilotApi initialization."""

    def test_init_stores_params(self, copilot_api):
        assert copilot_api.system_prompt == "You are a test assistant"
        assert copilot_api.model_name == "gpt-4.1"
        assert copilot_api.max_retries == 3
        assert copilot_api.retries == 0
        assert copilot_api.messages == []

    def test_init_starts_client_and_creates_session(self, mock_copilot_client):
        with patch("microbots.llm.copilot_api.CopilotClient", return_value=mock_copilot_client):
            api = CopilotApi(system_prompt="test", model_name="gpt-4.1")
            try:
                mock_copilot_client.start.assert_awaited_once()
                mock_copilot_client.create_session.assert_awaited_once()

                call_kwargs = mock_copilot_client.create_session.call_args[1]
                assert call_kwargs["model"] == "gpt-4.1"
                assert call_kwargs["infinite_sessions"] == {"enabled": False}
                assert "system_message" in call_kwargs
                assert call_kwargs["system_message"]["content"] == "test"
            finally:
                api.close()

    def test_implements_llm_interface(self, copilot_api):
        assert isinstance(copilot_api, LLMInterface)


@pytest.mark.unit
class TestCopilotApiAsk:
    """Tests for CopilotApi.ask() method."""

    def test_ask_returns_valid_response(self, copilot_api):
        response = copilot_api.ask("What files are in the directory?")

        assert isinstance(response, LLMAskResponse)
        assert response.task_done is False
        assert response.thoughts == "Thinking about the task"
        assert response.command == "ls -la"

    def test_ask_appends_to_messages(self, copilot_api):
        copilot_api.ask("test message")

        assert len(copilot_api.messages) == 2
        assert copilot_api.messages[0]["role"] == "user"
        assert copilot_api.messages[0]["content"] == "test message"
        assert copilot_api.messages[1]["role"] == "assistant"

    def test_ask_handles_task_done(self, copilot_api, mock_copilot_session):
        """Test ask when LLM signals task completion."""
        done_response = Mock()
        done_response.data.content = json.dumps({
            "task_done": True,
            "thoughts": "Task is complete",
            "command": ""
        })
        mock_copilot_session.send_and_wait = AsyncMock(return_value=done_response)

        response = copilot_api.ask("done?")
        assert response.task_done is True
        assert response.command == ""

    def test_ask_handles_markdown_wrapped_json(self, copilot_api, mock_copilot_session):
        """Test that JSON wrapped in markdown code blocks is extracted."""
        md_response = Mock()
        md_response.data.content = '```json\n{"task_done": false, "thoughts": "extracted", "command": "pwd"}\n```'
        mock_copilot_session.send_and_wait = AsyncMock(return_value=md_response)

        response = copilot_api.ask("test")
        assert response.thoughts == "extracted"
        assert response.command == "pwd"


@pytest.mark.unit
class TestCopilotApiClearHistory:
    """Tests for CopilotApi.clear_history() method."""

    def test_clear_history_resets_messages(self, copilot_api):
        copilot_api.messages = [{"role": "user", "content": "test"}]
        result = copilot_api.clear_history()

        assert result is True
        assert copilot_api.messages == []

    def test_clear_history_recreates_session(self, copilot_api, mock_copilot_session, mock_copilot_client):
        copilot_api.clear_history()

        mock_copilot_session.disconnect.assert_awaited()
        # create_session called once at init, once on clear_history
        assert mock_copilot_client.create_session.await_count == 2


@pytest.mark.unit
class TestCopilotApiClose:
    """Tests for CopilotApi.close() method."""

    def test_close_stops_client(self, mock_copilot_client, mock_copilot_session):
        with patch("microbots.llm.copilot_api.CopilotClient", return_value=mock_copilot_client):
            api = CopilotApi(system_prompt="test", model_name="gpt-4.1")
            api.close()

            mock_copilot_session.disconnect.assert_awaited()
            mock_copilot_client.stop.assert_awaited()


@pytest.mark.unit
class TestCopilotApiImportError:
    """Test that a helpful error is raised when ghcp extra is not installed."""

    def test_microbot_raises_helpful_error_without_ghcp(self):
        """MicroBot._create_llm() should raise ValueError when copilot SDK is missing."""
        from microbots.constants import ModelProvider

        with patch("microbots.MicroBot.ModelProvider", ModelProvider):
            # Simulate ImportError when trying to import CopilotApi
            with patch.dict("sys.modules", {"microbots.llm.copilot_api": None}):
                from microbots.MicroBot import MicroBot
                with pytest.raises(ValueError, match="pip install microbots\\[ghcp\\]"):
                    MicroBot(
                        model="github-copilot/gpt-4.1",
                        system_prompt="test",
                    )
