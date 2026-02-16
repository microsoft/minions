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
from microbots.llm.anthropic_api import AnthropicApi, DEFAULT_CONTEXT_MANAGEMENT


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
            f"{DOCKER_WORKING_DIR}/{base_name}",
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
            task, max_iterations=25, timeout_in_seconds=180,
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
            "dependencies."
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
            "What project name and dependencies were found in the "
            "previous analysis?"
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
            "Look at the Python source files in the repository. "
            "Identify the main entry point or purpose of the code."
        )

        response: BotRunResult = bot.run(
            task, max_iterations=25, timeout_in_seconds=240,
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


# ---------------------------------------------------------------------------
# Integration tests — memory only, context management only, both
# ---------------------------------------------------------------------------

_SKIP_NO_KEY = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


def _make_bot(model, test_repo, *, additional_tools=None, provider_options=None):
    """Helper: build a MicroBot with a sandbox and optional memory/cm config."""
    from microbots.MicroBot import system_prompt_common
    from microbots.constants import DOCKER_WORKING_DIR, PermissionLabels
    from microbots.extras.mount import Mount

    base_name = os.path.basename(str(test_repo))
    folder_mount = Mount(
        str(test_repo),
        f"{DOCKER_WORKING_DIR}/{base_name}",
        PermissionLabels.READ_ONLY,
    )

    system_prompt = (
        f"{system_prompt_common}\n\n"
        "You are a test assistant with persistent memory.\n"
        f"The repository is mounted at {folder_mount.sandbox_path}.\n"
        "IMPORTANT: Always save your key findings to memory using the memory "
        "tool before completing any task.\n"
        "Focus only on files directly related to the task."
    )

    return MicroBot(
        model=model,
        system_prompt=system_prompt,
        additional_tools=additional_tools or [],
        folder_to_mount=folder_mount,
        provider_options=provider_options,
    )


def _list_memory_files(memory_tool):
    """Return non-hidden regular files under the memory directory."""
    if not memory_tool.memory_dir.exists():
        return []
    return [
        f for f in memory_tool.memory_dir.rglob("*")
        if f.is_file() and not f.name.startswith(".")
    ]


# ---- 1. Memory tool only (no context management) --------------------------

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.docker
@_SKIP_NO_KEY
class TestBotMemoryOnly:
    """
    End-to-end: bot uses the memory tool WITHOUT context management.

    Proves:
    - The tool_runner loop actually calls the memory tool
    - The model writes data that lands on disk
    - A second bot can read it back (cross-session persistence)
    """

    @pytest.fixture
    def memory_tool(self, tmp_path):
        tool = AnthropicMemoryTool(memory_dir=tmp_path / "mem_only")
        yield tool
        tool.clear_all()

    def test_memory_tool_saves_to_disk(self, test_repo, memory_tool):
        """Bot with memory (no CM) writes findings to disk."""
        model = f"anthropic/{os.getenv('ANTHROPIC_DEPLOYMENT_NAME', 'claude-sonnet-4-5')}"
        bot = _make_bot(
            model, test_repo,
            additional_tools=[memory_tool],
            # NO provider_options → no context management
        )

        result: BotRunResult = bot.run(
            "List the files in the repository root and describe the project structure.",
            max_iterations=15,
            timeout_in_seconds=120,
        )

        print(f"Status={result.status}  Result={result.result}  Error={result.error}")
        assert result.status, f"Bot failed: {result.error}"

        files = _list_memory_files(memory_tool)
        print(f"Memory files: {[f.name for f in files]}")
        assert len(files) > 0, (
            "Memory tool was never called — no files on disk. "
            "The system prompt instructs the model to save findings."
        )

    def test_memory_persists_without_context_management(self, test_repo, memory_tool):
        """
        Run 1 saves a fact to memory.
        Run 2 (new bot, same memory dir) reads it back.
        No context management involved.
        """
        model = f"anthropic/{os.getenv('ANTHROPIC_DEPLOYMENT_NAME', 'claude-sonnet-4-5')}"

        # Run 1 — analyze (system prompt drives memory usage)
        bot1 = _make_bot(model, test_repo, additional_tools=[memory_tool])
        r1 = bot1.run(
            "Read pyproject.toml and identify the project name.",
            max_iterations=15,
            timeout_in_seconds=120,
        )
        assert r1.status, f"Run 1 failed: {r1.error}"
        assert len(_list_memory_files(memory_tool)) > 0, "Run 1 wrote nothing"

        # Run 2 — new bot, same memory dir. Can only know the answer via memory.
        bot2 = _make_bot(model, test_repo, additional_tools=[memory_tool])
        r2 = bot2.run(
            "What project name was found in the previous analysis?",
            max_iterations=10,
            timeout_in_seconds=90,
        )
        assert r2.status, f"Run 2 failed: {r2.error}"
        # The fixture test_repo has pyproject.toml with name = "testpkg"
        assert "testpkg" in (r2.result or "").lower(), (
            f"Run 2 didn't recall the project name. Got: {r2.result}"
        )


# ---- 2. Context management only (no memory tool) --------------------------

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.docker
@_SKIP_NO_KEY
class TestBotContextManagementOnly:
    """
    End-to-end: bot uses context management WITHOUT the memory tool.

    Proves:
    - The context_management config is accepted by the real Anthropic API
    - The correct beta header is sent (otherwise the API would reject it)
    - The bot completes a task successfully with CM enabled
    """

    def test_context_management_explicit_config(self, test_repo):
        """Bot with explicit clear_tool_uses context management config completes a task."""
        model = f"anthropic/{os.getenv('ANTHROPIC_DEPLOYMENT_NAME', 'claude-sonnet-4-5')}"
        # We still need an external tool for the tool_runner path.
        # Use a memory tool as a simple external tool but don't assert on memory usage.
        from tempfile import mkdtemp
        tmp = mkdtemp()
        mem = AnthropicMemoryTool(memory_dir=os.path.join(tmp, "mem_cm"))

        cm_config = {"edits": [{"type": "clear_tool_uses_20250919"}]}

        bot = _make_bot(
            model, test_repo,
            additional_tools=[mem],
            provider_options={"context_management": cm_config},
        )

        # Verify the LLM layer received the config
        assert bot.llm.context_management == cm_config
        betas = bot.llm._collect_betas()
        assert "context-management-2025-06-27" in betas, (
            f"Expected CM beta header in {betas}"
        )

        # Actually run the bot — if beta header or CM format is wrong, API raises
        result: BotRunResult = bot.run(
            "Run 'echo hello' and report the output.",
            max_iterations=10,
            timeout_in_seconds=90,
        )

        print(f"Status={result.status}  Result={result.result}  Error={result.error}")
        assert result.status, f"Bot failed with context management: {result.error}"
        assert result.result is not None
        mem.clear_all()

    def test_context_management_true_uses_default(self, test_repo):
        """Passing provider_options={'context_management': True} resolves to the default."""
        model = f"anthropic/{os.getenv('ANTHROPIC_DEPLOYMENT_NAME', 'claude-sonnet-4-5')}"
        from tempfile import mkdtemp
        tmp = mkdtemp()
        mem = AnthropicMemoryTool(memory_dir=os.path.join(tmp, "mem_cm_true"))

        bot = _make_bot(
            model, test_repo,
            additional_tools=[mem],
            provider_options={"context_management": True},
        )

        # Should have resolved to the default
        assert bot.llm.context_management == DEFAULT_CONTEXT_MANAGEMENT

        result: BotRunResult = bot.run(
            "Run 'echo default-cm-test' and report what you see.",
            max_iterations=10,
            timeout_in_seconds=90,
        )

        print(f"Status={result.status}  Result={result.result}  Error={result.error}")
        assert result.status, f"Bot failed with context_management=True: {result.error}"
        mem.clear_all()


# ---- 3. Both memory tool AND context management ---------------------------

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.docker
@_SKIP_NO_KEY
class TestBotMemoryWithContextManagement:
    """
    End-to-end: bot uses BOTH memory tool and context management together.

    Proves:
    - The beta header is not duplicated (memory tool + CM share the same one)
    - The API accepts the combined config without errors
    - The memory tool still works correctly when CM is also active
    - Data written to memory persists and can be read back
    """

    @pytest.fixture
    def memory_tool(self, tmp_path):
        tool = AnthropicMemoryTool(memory_dir=tmp_path / "mem_both")
        yield tool
        tool.clear_all()

    def test_no_beta_duplication(self, test_repo, memory_tool):
        """Memory tool and CM map to the same beta — must not duplicate."""
        model = f"anthropic/{os.getenv('ANTHROPIC_DEPLOYMENT_NAME', 'claude-sonnet-4-5')}"
        cm_config = {"edits": [{"type": "clear_tool_uses_20250919"}]}

        bot = _make_bot(
            model, test_repo,
            additional_tools=[memory_tool],
            provider_options={"context_management": cm_config},
        )

        betas = bot.llm._collect_betas()
        assert betas.count("context-management-2025-06-27") == 1, (
            f"Beta header duplicated: {betas}"
        )

    def test_memory_works_with_context_management(self, test_repo, memory_tool):
        """Bot with both memory and CM can save data and complete a task."""
        model = f"anthropic/{os.getenv('ANTHROPIC_DEPLOYMENT_NAME', 'claude-sonnet-4-5')}"
        cm_config = {"edits": [{"type": "clear_tool_uses_20250919"}]}

        bot = _make_bot(
            model, test_repo,
            additional_tools=[memory_tool],
            provider_options={"context_management": cm_config},
        )

        result: BotRunResult = bot.run(
            "List the repository files and describe the project structure.",
            max_iterations=25,
            timeout_in_seconds=180,
        )

        print(f"Status={result.status}  Result={result.result}  Error={result.error}")
        assert result.status, f"Bot failed: {result.error}"

        files = _list_memory_files(memory_tool)
        print(f"Memory files: {[f.name for f in files]}")
        assert len(files) > 0, (
            "Memory tool was never called despite system prompt instruction."
        )

    def test_memory_persists_with_context_management(self, test_repo, memory_tool):
        """
        Two-phase test with both features active.
        Run 1: save fact.  Run 2 (new bot): read it back from memory.
        """
        model = f"anthropic/{os.getenv('ANTHROPIC_DEPLOYMENT_NAME', 'claude-sonnet-4-5')}"
        cm_config = {"edits": [{"type": "clear_tool_uses_20250919"}]}

        # Run 1 — analyze (system prompt drives memory usage)
        bot1 = _make_bot(
            model, test_repo,
            additional_tools=[memory_tool],
            provider_options={"context_management": cm_config},
        )
        r1 = bot1.run(
            "Read pyproject.toml and identify the project name.",
            max_iterations=15,
            timeout_in_seconds=120,
        )
        assert r1.status, f"Run 1 failed: {r1.error}"

        files = _list_memory_files(memory_tool)
        assert len(files) > 0, "Run 1 wrote nothing to memory"
        for f in files:
            print(f"  {f.name}: {f.read_text()[:200]}")

        # Run 2 — new bot, same memory dir + CM. Can only know via memory.
        bot2 = _make_bot(
            model, test_repo,
            additional_tools=[memory_tool],
            provider_options={"context_management": cm_config},
        )
        r2 = bot2.run(
            "What project name was found in the previous analysis?",
            max_iterations=10,
            timeout_in_seconds=90,
        )
        assert r2.status, f"Run 2 failed: {r2.error}"
        assert "testpkg" in (r2.result or "").lower(), (
            f"Run 2 couldn't recall the project name. Got: {r2.result}"
        )
