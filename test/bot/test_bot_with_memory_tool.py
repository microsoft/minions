"""
Integration tests for creating a bot with the Anthropic memory tool.

These tests verify that:
1. A ReadingBot / MicroBot can be instantiated with the AnthropicMemoryTool.
2. The memory tool is correctly wired through to the AnthropicApi layer.
3. End-to-end: a bot can use the memory tool to persist and recall information
   while performing a task inside a sandboxed environment.
"""

import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

import logging

logging.basicConfig(level=logging.INFO)

from microbots import ReadingBot, MicroBot, BotRunResult
from microbots.MicroBot import BotType
from microbots.tools.external.anthropic_memory_tool import (
    AnthropicMemoryTool,
    MEMORY_BETA_HEADER,
)
from microbots.tools.external_tool import ExternalTool
from microbots.llm.anthropic_api import AnthropicApi


# ---------------------------------------------------------------------------
# Unit-level wiring tests (no API calls, no Docker)
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.docker
class TestBotMemoryToolWiring:
    """Verify that bots correctly separate and route the memory tool.

    These tests never call the LLM API — they only check that the memory tool
    is wired into the correct internal lists.  The Anthropic client object is
    created but never used, so no API key is needed.  Docker IS required
    because ReadingBot spins up a LocalDockerEnvironment on init.
    """

    @pytest.fixture
    def memory_tool(self, tmp_path):
        return AnthropicMemoryTool(memory_dir=tmp_path / "mem")

    def test_memory_tool_is_external_tool(self, memory_tool):
        """AnthropicMemoryTool must be recognised as an ExternalTool."""
        assert isinstance(memory_tool, ExternalTool)

    def test_reading_bot_separates_memory_tool(self, memory_tool, test_repo):
        """ReadingBot should place memory tool in _external_tools, not _internal_tools."""
        model = f"anthropic/{os.getenv('ANTHROPIC_DEPLOYMENT_NAME', 'claude-sonnet-4-5')}"
        bot = ReadingBot(
            model=model,
            folder_to_mount=str(test_repo),
            additional_tools=[memory_tool],
        )
        assert memory_tool in bot._external_tools
        assert memory_tool not in bot._internal_tools

    def test_microbot_separates_memory_tool(self, memory_tool, test_repo):
        """MicroBot should place memory tool in _external_tools, not _internal_tools."""
        model = f"anthropic/{os.getenv('ANTHROPIC_DEPLOYMENT_NAME', 'claude-sonnet-4-5')}"
        bot = MicroBot(
            model=model,
            additional_tools=[memory_tool],
        )
        assert memory_tool in bot._external_tools
        assert memory_tool not in bot._internal_tools

    def test_llm_receives_external_tools(self, memory_tool, test_repo):
        """The AnthropicApi instance created by the bot should hold the memory tool."""
        model = f"anthropic/{os.getenv('ANTHROPIC_DEPLOYMENT_NAME', 'claude-sonnet-4-5')}"
        bot = ReadingBot(
            model=model,
            folder_to_mount=str(test_repo),
            additional_tools=[memory_tool],
        )
        assert isinstance(bot.llm, AnthropicApi)
        assert memory_tool in bot.llm.external_tools

    def test_beta_header_collected(self, memory_tool, test_repo):
        """AnthropicApi._collect_betas() should include the memory beta header."""
        model = f"anthropic/{os.getenv('ANTHROPIC_DEPLOYMENT_NAME', 'claude-sonnet-4-5')}"
        bot = ReadingBot(
            model=model,
            folder_to_mount=str(test_repo),
            additional_tools=[memory_tool],
        )
        betas = bot.llm._collect_betas()
        assert MEMORY_BETA_HEADER in betas


# ---------------------------------------------------------------------------
# Integration test — live API + Docker sandbox
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.docker
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — Anthropic memory tool requires Anthropic API",
)
class TestBotMemoryToolIntegration:
    """End-to-end tests that hit the real Anthropic API and Docker environment.

    These require:
    - ANTHROPIC_API_KEY set in the environment
    - Docker daemon running
    - Network access to the Anthropic API

    NOTE: The Anthropic memory tool is Anthropic-specific. It uses
    ``beta.messages.tool_runner`` and cannot be used with Azure OpenAI or Ollama.
    """

    @pytest.fixture
    def memory_tool(self, tmp_path):
        tool = AnthropicMemoryTool(memory_dir=tmp_path / "mem_integration")
        yield tool
        tool.clear_all()

    def _make_memory_bot(self, model, test_repo, memory_tool):
        """
        Create a ReadingBot whose **system prompt** instructs it to always
        save important findings to memory.  This is how you'd configure the
        bot in production — the task itself never mentions memory.
        """
        from microbots.MicroBot import system_prompt_common
        from microbots.constants import DOCKER_WORKING_DIR, PermissionLabels
        from microbots.extras.mount import Mount
        import os as _os

        base_name = _os.path.basename(str(test_repo))
        folder_mount_info = Mount(
            str(test_repo),
            f"/{DOCKER_WORKING_DIR}/{base_name}",
            PermissionLabels.READ_ONLY,
        )

        system_prompt = (
            f"{system_prompt_common}\n\n"
            "You are a reading bot with persistent memory.\n"
            f"The repository is mounted at {folder_mount_info.sandbox_path}.\n"
            "IMPORTANT: Always save your key findings and analysis results to "
            "memory using the memory tool before completing any task. This "
            "ensures your work is preserved for future sessions.\n"
            "Do not explore unrelated files. Focus only on files directly "
            "related to the task."
        )

        return MicroBot(
            model=model,
            system_prompt=system_prompt,
            additional_tools=[memory_tool],
            folder_to_mount=folder_mount_info,
        )

    def _list_memory_files(self, memory_tool):
        """Return non-hidden files in the memory directory."""
        if not memory_tool.memory_dir.exists():
            return []
        return [f for f in memory_tool.memory_dir.iterdir()
                if not f.name.startswith(".") and f.is_file()]

    # ---- Test 1: System prompt nudge → model must write to memory ----------

    def test_bot_saves_findings_to_memory(self, test_repo, memory_tool):
        """
        The system prompt tells the bot to persist findings to memory.
        The task itself is a normal repo analysis — no mention of memory.
        We assert that at least one memory file was created.
        """
        model = f"anthropic/{os.getenv('ANTHROPIC_DEPLOYMENT_NAME', 'claude-sonnet-4-5')}"
        bot = self._make_memory_bot(model, test_repo, memory_tool)

        task = (
            "Analyze the repository structure. Identify the main programming "
            "language used and what the project does."
        )

        response: BotRunResult = bot.run(
            task, max_iterations=15, timeout_in_seconds=120,
        )

        print(f"Status: {response.status}, Result: {response.result}, Error: {response.error}")
        assert response.status, f"Bot failed: {response.error}"
        assert response.result is not None

        # ── This is the key assertion: memory MUST have been used ──
        memory_files = self._list_memory_files(memory_tool)
        print(f"Memory files created: {[f.name for f in memory_files]}")
        for mf in memory_files:
            print(f"  {mf.name}:\n{mf.read_text()[:300]}")

        assert len(memory_files) > 0, (
            "The model did NOT write anything to memory. "
            "The system prompt instructs it to save findings — check the prompt."
        )

    # ---- Test 2: Two-phase — run 2 can only pass if run 1 stored data ------

    def test_memory_persists_across_bot_runs(self, test_repo, memory_tool):
        """
        Run 1: Bot analyzes the repo and (via system prompt) saves findings.
        Run 2: A NEW bot, same memory dir, is asked what was found previously.
               It can only answer by reading from memory — it has no other way
               to know what the first bot discovered.
        """
        model = f"anthropic/{os.getenv('ANTHROPIC_DEPLOYMENT_NAME', 'claude-sonnet-4-5')}"

        # --- Run 1: Analyze and save ---
        bot1 = self._make_memory_bot(model, test_repo, memory_tool)

        task1 = (
            "Read the pyproject.toml and identify the project name and its "
            "dependencies. Save your findings."
        )

        result1: BotRunResult = bot1.run(
            task1, max_iterations=15, timeout_in_seconds=120,
        )
        assert result1.status, f"Run 1 failed: {result1.error}"

        # Verify something was stored
        files_after_run1 = self._list_memory_files(memory_tool)
        print(f"After run 1: {len(files_after_run1)} memory file(s): "
              f"{[f.name for f in files_after_run1]}")
        assert len(files_after_run1) > 0, "Run 1 did not save anything to memory"

        # Print what was stored so we can see it in the logs
        for mf in files_after_run1:
            print(f"  {mf.name}:\n{mf.read_text()[:300]}")

        # --- Run 2: New bot, only source of info is memory ---
        bot2 = self._make_memory_bot(model, test_repo, memory_tool)

        task2 = (
            "Check your memory for any previous analysis notes. "
            "What project name and dependencies were found? "
            "Only look at memory, do not re-read the repository files."
        )

        result2: BotRunResult = bot2.run(
            task2, max_iterations=10, timeout_in_seconds=90,
        )

        print(f"Run 2 result: {result2.result}")
        assert result2.status, f"Run 2 failed: {result2.error}"
        assert result2.result is not None
        # The project is "testpkg" with "numpy" dependency per pyproject.toml
        result_lower = result2.result.lower()
        assert "testpkg" in result_lower or "numpy" in result_lower, (
            f"Run 2 could not recall findings from run 1. Got: {result2.result}"
        )

    # ---- Test 3: Bug investigation with memory tracking --------------------

    def test_bot_tracks_bugs_in_memory(self, test_repo, memory_tool):
        """
        Give the bot a debugging task with system prompt that says to track
        findings in memory. Assert that memory was used.
        """
        model = f"anthropic/{os.getenv('ANTHROPIC_DEPLOYMENT_NAME', 'claude-sonnet-4-5')}"
        bot = self._make_memory_bot(model, test_repo, memory_tool)

        task = (
            "Look at the Python source files and test files in the repository. "
            "Identify any bugs or issues. Focus only on Python files."
        )

        response: BotRunResult = bot.run(
            task, max_iterations=20, timeout_in_seconds=180,
        )

        print(f"Status: {response.status}, Result: {response.result}, Error: {response.error}")
        assert response.status, f"Bot failed: {response.error}"
        assert response.result is not None

        memory_files = self._list_memory_files(memory_tool)
        print(f"Memory files: {[f.name for f in memory_files]}")
        for mf in memory_files:
            print(f"  {mf.name}:\n{mf.read_text()[:300]}")

        assert len(memory_files) > 0, (
            "The model did NOT save bug findings to memory."
        )


# ---------------------------------------------------------------------------
# Custom MicroBot with memory tool
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — Anthropic memory tool requires Anthropic API",
)
class TestCustomBotWithMemoryTool:
    """Test using MicroBot directly (not a specialised bot) with the memory tool."""

    @pytest.fixture
    def memory_tool(self, tmp_path):
        tool = AnthropicMemoryTool(memory_dir=tmp_path / "mem_custom")
        yield tool
        tool.clear_all()

    def test_custom_bot_with_memory_tool(self, memory_tool):
        """A MicroBot with memory-aware system prompt saves findings."""
        from microbots.MicroBot import system_prompt_common

        model = f"anthropic/{os.getenv('ANTHROPIC_DEPLOYMENT_NAME', 'claude-sonnet-4-5')}"

        custom_prompt = (
            f"{system_prompt_common}\n\n"
            "You are a code review assistant with persistent memory.\n"
            "IMPORTANT: Always save your observations and results to memory "
            "using the memory tool before completing a task."
        )

        bot = MicroBot(
            model=model,
            bot_type=BotType.CUSTOM_BOT,
            system_prompt=custom_prompt,
            additional_tools=[memory_tool],
        )

        task = (
            "Run 'echo hello world' and 'date'. Report what you see."
        )

        response: BotRunResult = bot.run(
            task, max_iterations=10, timeout_in_seconds=90,
        )

        print(f"Status: {response.status}, Result: {response.result}, Error: {response.error}")
        assert response.status, f"Bot failed: {response.error}"
        assert response.result is not None

        # Verify memory was used
        memory_files = [f for f in memory_tool.memory_dir.iterdir()
                        if not f.name.startswith(".") and f.is_file()]
        print(f"Memory files: {[f.name for f in memory_files]}")
        for mf in memory_files:
            print(f"  {mf.name}:\n{mf.read_text()[:200]}")

        assert len(memory_files) > 0, (
            "The model did NOT write to memory despite system prompt instruction."
        )
