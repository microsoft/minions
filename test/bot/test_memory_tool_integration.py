"""Tests for the Anthropic memory tool end-to-end flow.

Unit tests (mocked API):
  Verify wiring — auto-upgrade, tool dispatch, and memory file operations
  with a mocked Anthropic client. Fast, free, no API key needed.

Integration tests (real API):
  Hit the actual Anthropic API to verify the full round-trip.
  Gated behind ``@pytest.mark.anthropic_integration``.
  Require ``ANTHROPIC_API_KEY`` in .env.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

from microbots import MicroBot, BotRunResult
from microbots.llm.llm import llm_output_format_str
from microbots.tools.tool_definitions.memory_tool import MemoryTool
from microbots.tools.tool_definitions.anthropic_memory_tool import AnthropicMemoryTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_use_response(tool_name, tool_id, tool_input, thinking_text=""):
    """Build a mock Anthropic API response with stop_reason='tool_use'."""
    blocks = []

    if thinking_text:
        text_block = Mock()
        text_block.type = "text"
        text_block.text = thinking_text
        blocks.append(text_block)

    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.name = tool_name
    tool_block.id = tool_id
    tool_block.input = tool_input
    tool_block.model_dump = Mock(return_value={
        "type": "tool_use",
        "id": tool_id,
        "name": tool_name,
        "input": tool_input,
    })
    blocks.append(tool_block)

    resp = Mock()
    resp.stop_reason = "tool_use"
    resp.content = blocks
    return resp


def _make_end_turn_response(task_done, thoughts, command=""):
    """Build a mock Anthropic API response with stop_reason='end_turn'."""
    payload = json.dumps({
        "task_done": task_done,
        "thoughts": thoughts,
        "command": command,
    })

    text_block = Mock()
    text_block.type = "text"
    text_block.text = payload

    resp = Mock()
    resp.stop_reason = "end_turn"
    resp.content = [text_block]
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMemoryToolWiring:
    """Unit tests — mocked Anthropic client, real tool dispatch and file ops."""

    @pytest.fixture()
    def memory_dir(self, tmp_path):
        d = tmp_path / "memory"
        d.mkdir()
        return d

    @pytest.fixture()
    def bot(self, memory_dir):
        """Create a MicroBot with Anthropic provider and a MemoryTool.

        The Anthropic client is mocked, but the rest of the stack is real:
        auto-upgrade, tool dispatch, and memory file operations.
        """
        tool = MemoryTool(
            memory_dir=str(memory_dir),
            usage_instructions_to_llm="Use the memory tool to persist notes.",
        )

        mock_env = Mock()
        mock_env.execute.return_value = Mock(return_code=0, stdout="", stderr="")

        anthropic_deployment = "claude-sonnet-4-5"

        with patch("microbots.llm.anthropic_api.Anthropic") as mock_anthropic_cls, \
             patch("microbots.llm.anthropic_api.api_key", "test-key"), \
             patch("microbots.llm.anthropic_api.endpoint", "https://api.anthropic.com"), \
             patch("microbots.llm.anthropic_api.deployment_name", anthropic_deployment):

            bot = MicroBot(
                model=f"anthropic/{anthropic_deployment}",
                system_prompt="You are a helpful assistant.",
                additional_tools=[tool],
                environment=mock_env,
            )

            self._mock_client = mock_anthropic_cls.return_value
            yield bot
            del bot

    # -- Upgrade verification -----------------------------------------------

    def test_memory_tool_auto_upgraded_to_anthropic_variant(self, bot):
        """MemoryTool passed to MicroBot should be auto-upgraded to AnthropicMemoryTool."""
        upgraded_tools = bot.additional_tools
        memory_tools = [t for t in upgraded_tools if isinstance(t, AnthropicMemoryTool)]
        assert len(memory_tools) == 1, "Expected exactly one AnthropicMemoryTool after auto-upgrade"

    def test_tool_dicts_include_memory_schema(self, bot):
        """The LLM should have received the memory tool schema."""
        assert len(bot.llm._tool_dicts) == 1
        assert bot.llm._tool_dicts[0]["type"] == "memory_20250818"

    # -- Create file via tool_use -------------------------------------------

    def test_create_memory_file_via_tool_dispatch(self, bot, memory_dir):
        """LLM requests a memory create → MicroBot dispatches → file appears on disk."""
        # Sequence:
        # 1. ask(task) → API returns tool_use (memory create)
        # 2. ask(tool_result) → API returns end_turn (task_done=True)
        self._mock_client.messages.create.side_effect = [
            _make_tool_use_response(
                tool_name="memory",
                tool_id="tool_001",
                tool_input={
                    "command": "create",
                    "path": "/memories/notes.md",
                    "file_text": "Hello from integration test",
                },
                thinking_text="I'll save a note to memory.",
            ),
            _make_end_turn_response(
                task_done=True,
                thoughts="Saved a note to memory successfully.",
            ),
        ]

        result: BotRunResult = bot.run(
            "Save a note saying 'Hello from integration test'",
            max_iterations=5,
            timeout_in_seconds=30,
        )

        assert result.status is True
        assert result.error is None

        # Verify the file was actually created on disk
        # _resolve("/memories/notes.md") strips the "memories/" prefix → memory_dir/notes.md
        created_file = memory_dir / "notes.md"
        assert created_file.exists(), f"Expected {created_file} to be created"
        assert created_file.read_text() == "Hello from integration test"

    # -- View file via tool_use ---------------------------------------------

    def test_view_memory_file_via_tool_dispatch(self, bot, memory_dir):
        """LLM requests a memory view → MicroBot dispatches → file content returned."""
        # Pre-create a file in memory
        # _resolve("/memories/existing.md") → memory_dir/existing.md
        (memory_dir / "existing.md").write_text("Previously saved content")

        self._mock_client.messages.create.side_effect = [
            _make_tool_use_response(
                tool_name="memory",
                tool_id="tool_002",
                tool_input={
                    "command": "view",
                    "path": "/memories/existing.md",
                },
                thinking_text="Let me check my memory.",
            ),
            _make_end_turn_response(
                task_done=True,
                thoughts="Found previously saved content in memory.",
            ),
        ]

        result: BotRunResult = bot.run(
            "Check your memory for existing notes",
            max_iterations=5,
            timeout_in_seconds=30,
        )

        assert result.status is True

        # Verify the view result was passed back to the API as tool_result
        calls = self._mock_client.messages.create.call_args_list
        assert len(calls) == 2
        # The second call should have messages including the tool_result
        second_call_messages = calls[1].kwargs.get("messages") or calls[1][1].get("messages", [])
        tool_result_msgs = [
            m for m in second_call_messages
            if m.get("role") == "user" and isinstance(m.get("content"), list)
            and any(c.get("type") == "tool_result" for c in m["content"])
        ]
        assert len(tool_result_msgs) >= 1, "Expected a tool_result message in the second API call"
        # The tool_result content should contain the file content
        tool_result_content = tool_result_msgs[-1]["content"][0]["content"]
        assert "Previously saved content" in tool_result_content

    # -- Multiple tool calls in sequence ------------------------------------

    def test_create_then_view_memory_file(self, bot, memory_dir):
        """LLM creates a file, then views it — both dispatched via MicroBot loop."""
        self._mock_client.messages.create.side_effect = [
            # Step 1: create file
            _make_tool_use_response(
                tool_name="memory",
                tool_id="tool_003",
                tool_input={
                    "command": "create",
                    "path": "/memories/todo.md",
                    "file_text": "- Fix bug #42\n- Write tests",
                },
                thinking_text="Creating a todo list.",
            ),
            # Step 2: view file
            _make_tool_use_response(
                tool_name="memory",
                tool_id="tool_004",
                tool_input={
                    "command": "view",
                    "path": "/memories/todo.md",
                },
                thinking_text="Let me verify what I wrote.",
            ),
            # Step 3: done
            _make_end_turn_response(
                task_done=True,
                thoughts="Created and verified the todo list.",
            ),
        ]

        result: BotRunResult = bot.run(
            "Create a todo list and verify it was saved",
            max_iterations=10,
            timeout_in_seconds=30,
        )

        assert result.status is True
        assert result.error is None

        # File should exist with correct content
        created_file = memory_dir / "todo.md"
        assert created_file.exists()
        assert "Fix bug #42" in created_file.read_text()

    # -- Non-memory commands still go to environment ------------------------

    def test_non_memory_commands_go_to_environment(self, bot):
        """Regular shell commands should be dispatched to the environment, not the memory tool."""
        self._mock_client.messages.create.side_effect = [
            _make_end_turn_response(
                task_done=False,
                thoughts="Let me check the files.",
                command="ls -la",
            ),
            _make_end_turn_response(
                task_done=True,
                thoughts="Done.",
            ),
        ]

        result: BotRunResult = bot.run(
            "List the files",
            max_iterations=5,
            timeout_in_seconds=30,
        )

        assert result.status is True
        # The environment.execute should have been called with "ls -la"
        bot.environment.execute.assert_called_with("ls -la")


# ---------------------------------------------------------------------------
# Real integration tests — require ANTHROPIC_API_KEY
# ---------------------------------------------------------------------------

MEMORY_SYSTEM_PROMPT = f"""You are a helpful assistant with access to a memory tool.
You can save and retrieve notes using the memory tool.
All your responses must be in this JSON format:
{llm_output_format_str}
The properties (task_done, thoughts, command) are mandatory on each response.
When you are done, set task_done to true and command to an empty string.
"""


@pytest.mark.anthropic_integration
@pytest.mark.docker
class TestMemoryToolRealApi:
    """End-to-end integration tests that hit the real Anthropic API.

    These tests exercise the full MicroBot → AnthropicApi → memory tool
    pipeline with no mocking.  A real Docker environment is created
    (matching the AgentBoss integration test pattern).

    Run with::

        pytest -m anthropic_integration

    Requires ``ANTHROPIC_API_KEY`` in ``.env``.
    """

    @pytest.fixture()
    def memory_dir(self, tmp_path):
        d = tmp_path / "memory"
        d.mkdir()
        return d

    @pytest.fixture()
    def memory_bot(self, memory_dir):
        """Create a MicroBot with the real Anthropic API, real Docker env,
        and a MemoryTool.  No mocking — fully end-to-end.
        """
        tool = MemoryTool(
            memory_dir=str(memory_dir),
            usage_instructions_to_llm="Use the memory tool to persist notes.",
        )

        anthropic_deployment = os.getenv("ANTHROPIC_DEPLOYMENT_NAME", "claude-sonnet-4-5")

        bot = MicroBot(
            model=f"anthropic/{anthropic_deployment}",
            system_prompt=MEMORY_SYSTEM_PROMPT,
            additional_tools=[tool],
        )

        yield bot
        del bot

    def test_memory_tool_auto_upgraded(self, memory_bot):
        """MemoryTool should be silently auto-upgraded to AnthropicMemoryTool."""
        memory_tools = [t for t in memory_bot.additional_tools if isinstance(t, AnthropicMemoryTool)]
        assert len(memory_tools) == 1, "Expected exactly one AnthropicMemoryTool after auto-upgrade"

    def test_create_memory_file(self, memory_bot, memory_dir):
        """MicroBot should persist a debugging plan to memory.

        The LLM is expected to:
          1. Receive a task about planning a debugging session.
          2. Decide to persist the plan using the memory tool.
          3. Confirm the task is done.

        We verify the plan was actually written to disk.
        """
        result: BotRunResult = memory_bot.run(
            task=(
                "You are investigating a bug where the server returns HTTP 500 "
                "on POST /api/users. Create a debugging plan that includes: "
                "1) check server logs, 2) reproduce the request with curl, "
                "3) inspect the database connection. "
                "Persist this plan so you can resume later if interrupted."
            ),
            max_iterations=10,
            timeout_in_seconds=60,
        )

        assert result.status is True, f"Task failed: {result.error}"
        assert result.error is None

        # The LLM should have used the memory tool to persist the plan
        saved_files = [f for f in memory_dir.rglob("*") if f.is_file()]
        assert len(saved_files) >= 1, (
            f"Expected at least one file created in memory. "
            f"Found: {saved_files}"
        )
        combined_content = "\n".join(f.read_text() for f in saved_files).lower()
        assert "log" in combined_content or "curl" in combined_content or "database" in combined_content, (
            f"Expected debugging plan content in memory files. Content: {combined_content}"
        )

    def test_create_and_view_roundtrip(self, memory_bot, memory_dir):
        """MicroBot should save findings and then review them before reporting.

        The LLM is expected to:
          1. Record analysis findings using the memory tool.
          2. Review what it recorded to verify nothing was missed.
          3. Summarize the findings in its final thoughts.

        We verify:
          - At least one file was written to disk.
          - The LLM's summary references the recorded findings.
        """
        result: BotRunResult = memory_bot.run(
            task=(
                "You analyzed a Python project and found these issues: "
                "1) an unused import 'os' in utils.py, "
                "2) a missing null check in handler.py line 42. "
                "Record these findings, then review your notes and "
                "summarize what you found in your final thoughts."
            ),
            max_iterations=15,
            timeout_in_seconds=60,
        )

        assert result.status is True, f"Task failed: {result.error}"
        assert result.error is None

        # The LLM should have created at least one memory file
        saved_files = [f for f in memory_dir.rglob("*") if f.is_file()]
        assert len(saved_files) >= 1, (
            f"Expected at least one file in memory. "
            f"Found: {list(memory_dir.rglob('*'))}"
        )

        result_lower = result.result.lower()
        assert "import" in result_lower or "null" in result_lower or "handler" in result_lower, (
            f"LLM should have summarized the findings. Got: {result.result}"
        )
