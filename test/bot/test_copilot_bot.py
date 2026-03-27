"""
Unit and integration tests for CopilotBot.

Unit tests mock the copilot SDK and Docker environment to verify the
wiring and lifecycle.  Integration tests (marked ``@pytest.mark.integration``)
require a real Docker daemon, copilot-cli, and GitHub authentication.
"""

import importlib
import os
import shutil
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

# ---------------------------------------------------------------------------
# Mock the copilot SDK before importing CopilotBot (optional dependency)
# ---------------------------------------------------------------------------
_mock_copilot = MagicMock()
_mock_copilot.CopilotClient = MagicMock
_mock_copilot.ExternalServerConfig = MagicMock

_mock_permission = MagicMock()
_mock_permission.PermissionHandler = MagicMock()
_mock_permission.PermissionHandler.approve_all = MagicMock()
_mock_permission.PermissionRequestResult = MagicMock

_mock_events = MagicMock()
_mock_events.SessionEventType = MagicMock()
_mock_events.SessionEventType.ASSISTANT_MESSAGE = "assistant.message"
_mock_events.SessionEventType.ASSISTANT_MESSAGE_DELTA = "assistant.message_delta"
_mock_events.SessionEventType.SESSION_IDLE = "session.idle"

_mock_tools = MagicMock()
_mock_tools.Tool = MagicMock
_mock_tools.ToolInvocation = MagicMock
_mock_tools.ToolResult = MagicMock
_mock_tools.define_tool = MagicMock

sys.modules.setdefault("copilot", _mock_copilot)
sys.modules.setdefault("copilot.session", _mock_permission)
sys.modules.setdefault("copilot.generated.session_events", _mock_events)
sys.modules.setdefault("copilot.tools", _mock_tools)
sys.modules.setdefault("copilot.types", MagicMock())

# Reload to pick up mock
if "microbots.bot.CopilotBot" in sys.modules:
    importlib.reload(sys.modules["microbots.bot.CopilotBot"])

from microbots.MicroBot import BotRunResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copilot_cli_available():
    return shutil.which("copilot") is not None


def _copilot_sdk_installed():
    try:
        import copilot  # noqa: F401
        return not isinstance(copilot, MagicMock)
    except ImportError:
        return False


def _copilot_auth_available():
    if os.environ.get("GITHUB_TOKEN") or os.environ.get("COPILOT_GITHUB_TOKEN"):
        return True
    if shutil.which("gh"):
        try:
            result = subprocess.run(
                ["gh", "auth", "status"], capture_output=True, timeout=10
            )
            return result.returncode == 0
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# Unit test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_environment():
    """Create a mock LocalDockerEnvironment."""
    env = MagicMock()
    env.port = 9000
    env.container_port = 8080
    env.container = MagicMock()
    env.container.id = "abc123def456"
    env.image = "kavyasree261002/shell_server:latest"
    env.working_dir = "/tmp/mock_workdir"
    env.folder_to_mount = None
    env.overlay_mount = False

    # Make execute return success by default
    success_return = MagicMock()
    success_return.return_code = 0
    success_return.stdout = "copilot version 1.0.0"
    success_return.stderr = ""
    env.execute = MagicMock(return_value=success_return)
    env.copy_to_container = MagicMock(return_value=True)
    env.stop = MagicMock()
    return env


@pytest.fixture
def mock_copilot_session():
    """Mock Copilot SDK session."""
    session = AsyncMock()
    session.disconnect = AsyncMock()

    response = Mock()
    response.data = Mock()
    response.data.content = "Task completed successfully."
    session.send_and_wait = AsyncMock(return_value=response)
    session.on = MagicMock()
    return session


@pytest.fixture
def mock_copilot_client(mock_copilot_session):
    """Mock CopilotClient."""
    client = AsyncMock()
    client.start = AsyncMock()
    client.stop = AsyncMock()
    client.create_session = AsyncMock(return_value=mock_copilot_session)
    return client


@pytest.fixture
def copilot_bot(mock_environment, mock_copilot_client):
    """Create a CopilotBot with all external dependencies mocked."""
    with (
        patch("microbots.bot.CopilotBot.LocalDockerEnvironment", return_value=mock_environment),
        patch("microbots.bot.CopilotBot.get_free_port", side_effect=[9000, 4322, 4323]),
        patch("microbots.bot.CopilotBot.CopilotBot._install_copilot_cli"),
        patch("microbots.bot.CopilotBot.CopilotBot._start_copilot_cli_server"),
        patch("microbots.bot.CopilotBot.CopilotBot._wait_for_cli_ready"),
        patch("copilot.CopilotClient", return_value=mock_copilot_client),
        patch("copilot.ExternalServerConfig", return_value=MagicMock()),
    ):
        from microbots.bot.CopilotBot import CopilotBot
        bot = CopilotBot(
            model="gpt-4.1",
            environment=mock_environment,
            github_token="ghp_test_token_123",
        )
        yield bot
        # Stop the event loop thread properly before teardown
        try:
            bot._loop.call_soon_threadsafe(bot._loop.stop)
            bot._thread.join(timeout=2)
        except Exception:
            pass
        bot.environment = None  # Prevent stop() from trying env.stop() again


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCopilotBotInit:
    """Tests for CopilotBot initialisation."""

    def test_stores_model(self, copilot_bot):
        assert copilot_bot.model == "gpt-4.1"

    def test_stores_github_token(self, copilot_bot):
        assert copilot_bot.github_token == "ghp_test_token_123"

    def test_environment_assigned(self, copilot_bot, mock_environment):
        assert copilot_bot.environment is mock_environment

    def test_additional_tools_default_empty(self, copilot_bot):
        assert copilot_bot.additional_tools == []

    def test_import_error_without_sdk(self):
        """CopilotBot raises ImportError when copilot SDK is not installed."""
        # Temporarily remove the mock so the import fails
        saved = sys.modules.get("copilot")
        try:
            sys.modules["copilot"] = None  # Force ImportError on import
            # Need to reload the module
            if "microbots.bot.CopilotBot" in sys.modules:
                importlib.reload(sys.modules["microbots.bot.CopilotBot"])
            from microbots.bot.CopilotBot import CopilotBot as CB
            with pytest.raises(ImportError, match="github-copilot-sdk"):
                CB(model="gpt-4.1")
        finally:
            sys.modules["copilot"] = saved
            if "microbots.bot.CopilotBot" in sys.modules:
                importlib.reload(sys.modules["microbots.bot.CopilotBot"])


@pytest.mark.unit
class TestCopilotBotRun:
    """Tests for CopilotBot.run()."""

    def test_run_returns_bot_run_result(self, copilot_bot):
        result = copilot_bot.run("Fix the bug in main.py")
        assert isinstance(result, BotRunResult)

    def test_run_success(self, copilot_bot):
        result = copilot_bot.run("Fix the bug in main.py")
        assert result.status is True
        assert result.error is None
        assert result.result is not None

    def test_run_calls_tool_setup(self, copilot_bot, mock_environment):
        mock_tool = MagicMock()
        mock_tool.usage_instructions_to_llm = "Use tool X"
        mock_tool.install_commands = []
        mock_tool.verify_commands = []
        copilot_bot.additional_tools = [mock_tool]

        copilot_bot.run("test task")
        mock_tool.setup_tool.assert_called_once_with(mock_environment)

    def test_run_handles_exception(self, copilot_bot):
        """Run returns failure BotRunResult on exceptions."""
        with patch.object(copilot_bot, "_run_async", side_effect=RuntimeError("boom")):
            result = copilot_bot.run("test")
            assert result.status is False
            assert "boom" in result.error


@pytest.mark.unit
class TestCopilotBotSystemMessage:
    """Tests for system message construction."""

    def test_system_message_empty_no_mount_no_tools(self, copilot_bot):
        msg = copilot_bot._build_system_message()
        assert msg == ""

    def test_system_message_includes_mount_path(self, mock_environment, mock_copilot_client):
        with (
            patch("microbots.bot.CopilotBot.LocalDockerEnvironment", return_value=mock_environment),
            patch("microbots.bot.CopilotBot.get_free_port", side_effect=[9000, 4322, 4323]),
            patch("microbots.bot.CopilotBot.CopilotBot._install_copilot_cli"),
            patch("microbots.bot.CopilotBot.CopilotBot._start_copilot_cli_server"),
            patch("microbots.bot.CopilotBot.CopilotBot._wait_for_cli_ready"),
            patch("copilot.CopilotClient", return_value=mock_copilot_client),
            patch("copilot.ExternalServerConfig", return_value=MagicMock()),
            patch("microbots.bot.CopilotBot.CopilotBot._map_cli_port"),
            patch("microbots.bot.CopilotBot.CopilotBot._create_environment"),
        ):
            from microbots.bot.CopilotBot import CopilotBot
            from microbots.extras.mount import Mount
            mount = Mount("/tmp/test_repo", "/workdir/test_repo", "READ_WRITE")
            bot = CopilotBot(
                model="gpt-4.1",
                environment=mock_environment,
                github_token="ghp_test",
            )
            bot.folder_to_mount = mount
            msg = bot._build_system_message()
            assert "/workdir/test_repo" in msg
            bot.stop()

    def test_system_message_includes_tool_instructions(self, copilot_bot):
        mock_tool = MagicMock()
        mock_tool.usage_instructions_to_llm = "# Use browser command"
        copilot_bot.additional_tools = [mock_tool]

        msg = copilot_bot._build_system_message()
        assert "browser" in msg


@pytest.mark.unit
class TestCopilotBotStop:
    """Tests for CopilotBot.stop()."""

    def test_stop_cleans_environment(self, copilot_bot, mock_environment):
        copilot_bot.stop()
        mock_environment.stop.assert_called_once()

    def test_stop_idempotent(self, copilot_bot, mock_environment):
        copilot_bot.stop()
        copilot_bot.stop()  # Should not raise


@pytest.mark.unit
class TestCopilotBotCLIInstall:
    """Tests for copilot-cli installation logic."""

    def test_install_cli_calls_execute(self, mock_environment):
        from microbots.bot.CopilotBot import CopilotBot

        with (
            patch("microbots.bot.CopilotBot.get_free_port", side_effect=[9000, 4322]),
            patch("microbots.bot.CopilotBot.CopilotBot._start_copilot_cli_server"),
            patch("microbots.bot.CopilotBot.CopilotBot._wait_for_cli_ready"),
            patch("copilot.CopilotClient", return_value=AsyncMock()),
            patch("copilot.ExternalServerConfig", return_value=MagicMock()),
        ):
            bot = CopilotBot(
                model="gpt-4.1",
                environment=mock_environment,
                github_token="ghp_test",
            )
            # _install_copilot_cli was called during __init__
            # Verify that execute was called with npm install command
            calls = [str(c) for c in mock_environment.execute.call_args_list]
            npm_calls = [c for c in calls if "npm install" in c or "copilot" in c]
            assert len(npm_calls) > 0, "Expected copilot-cli install commands"
            bot.stop()

    def test_install_cli_raises_on_failure(self, mock_environment):
        from microbots.bot.CopilotBot import CopilotBot

        fail_return = MagicMock()
        fail_return.return_code = 1
        fail_return.stdout = ""
        fail_return.stderr = "npm ERR! not found"
        mock_environment.execute = MagicMock(return_value=fail_return)

        with (
            patch("microbots.bot.CopilotBot.get_free_port", side_effect=[9000, 4322]),
            patch("microbots.bot.CopilotBot.CopilotBot._start_copilot_cli_server"),
            patch("microbots.bot.CopilotBot.CopilotBot._wait_for_cli_ready"),
            patch("copilot.CopilotClient", return_value=AsyncMock()),
            patch("copilot.ExternalServerConfig", return_value=MagicMock()),
        ):
            with pytest.raises(RuntimeError, match="Failed to install copilot-cli"):
                CopilotBot(
                    model="gpt-4.1",
                    environment=mock_environment,
                    github_token="ghp_test",
                )


# ---------------------------------------------------------------------------
# Integration tests — require real Docker + copilot-cli + auth
# ---------------------------------------------------------------------------

_skip_no_copilot_cli = pytest.mark.skipif(
    not _copilot_cli_available(),
    reason="GitHub Copilot CLI not installed (copilot not in PATH)",
)

_skip_no_copilot_sdk = pytest.mark.skipif(
    not _copilot_sdk_installed(),
    reason="github-copilot-sdk not installed (pip install microbots[ghcp])",
)

_skip_no_copilot_auth = pytest.mark.skipif(
    not _copilot_auth_available(),
    reason="No GitHub auth available (set GITHUB_TOKEN or run 'gh auth login')",
)


@_skip_no_copilot_cli
@_skip_no_copilot_sdk
@_skip_no_copilot_auth
@pytest.mark.integration
@pytest.mark.slow
class TestCopilotBotIntegration:
    """End-to-end integration tests with real Copilot SDK."""

    def test_simple_task(self, test_repo, issue_1):
        """CopilotBot can fix a simple syntax error."""
        from microbots.bot.CopilotBot import CopilotBot

        issue_text = issue_1[0]
        verify_function = issue_1[1]

        bot = CopilotBot(
            model="gpt-4.1",
            folder_to_mount=str(test_repo),
            permission="READ_WRITE",
        )

        try:
            result = bot.run(
                issue_text,
                timeout_in_seconds=300,
            )
            assert result.status is True, f"CopilotBot failed: {result.error}"
            verify_function(test_repo)
        finally:
            bot.stop()
