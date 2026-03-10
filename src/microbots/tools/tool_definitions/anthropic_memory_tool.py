"""
AnthropicMemoryTool — wraps Anthropic's memory tool.

The memory tool lets the model persist information across conversations by
reading and writing files in a local memory directory.  When the model invokes
the tool, it sends a command (view, create, str_replace, insert, delete,
rename) and the client executes it against a local filesystem directory.

This implementation extends both:
  - ``MemoryTool``:  provides all file-operation logic (_resolve, _view,
    _create, _str_replace, _insert, _delete, _rename, _clear) and satisfies
    the ``ToolAbstract`` ABC (install_tool, verify_tool_installation, etc.).
  - ``BetaAbstractMemoryTool`` (SDK): provides native Anthropic dispatch and
    the ``to_dict()`` / ``call()`` interface required by AnthropicApi.

The SDK command-handler overrides (view, create, str_replace, insert, delete,
rename) simply translate SDK command objects → arg lists and delegate to the
inherited MemoryTool private methods, converting the CmdReturn back to a
string as the SDK expects.

The memory tool (type ``memory_20250818``) is available in the standard
Anthropic library and does not require a beta endpoint or header.  Pass it
via ``tools=[{"type": "memory_20250818", "name": "memory"}]`` on a regular
``client.messages.create(...)`` call.  ``MicroBot`` auto-upgrades
``MemoryTool`` to ``AnthropicMemoryTool`` for Anthropic providers and
passes the tool schema to ``AnthropicApi`` via ``tool_dicts``.

Usage:
    from microbots.tools.tool_definitions.anthropic_memory_tool import AnthropicMemoryTool

    memory = AnthropicMemoryTool()
    bot = ReadingBot(..., additional_tools=[memory])
"""

from __future__ import annotations

import json
from logging import getLogger
from pathlib import Path

from typing_extensions import override

from anthropic.lib.tools import BetaAbstractMemoryTool as _SDKMemoryTool
from anthropic.types.beta import (
    BetaMemoryTool20250818CreateCommand,
    BetaMemoryTool20250818DeleteCommand,
    BetaMemoryTool20250818InsertCommand,
    BetaMemoryTool20250818RenameCommand,
    BetaMemoryTool20250818StrReplaceCommand,
    BetaMemoryTool20250818ViewCommand,
)

from microbots.environment.Environment import CmdReturn
from microbots.tools.tool_definitions.memory_tool import MemoryTool

logger = getLogger(__name__)

DEFAULT_MEMORY_INSTRUCTIONS = (
    "MEMORY PROTOCOL:\n"
    "1. ALWAYS view your memory directory BEFORE doing anything else "
    "using the `view` command of your `memory` tool to check for earlier progress.\n"
    "2. As you make progress on the task, record status, progress, "
    "and key findings in your memory using the memory tool.\n"
    "3. ASSUME INTERRUPTION: Your context window might be reset at any moment, "
    "so you risk losing any progress that is not recorded in your memory directory.\n"
    "4. Before completing a task, always save your final results and analysis to memory.\n"
    "5. When editing your memory folder, always keep its content up-to-date, coherent "
    "and organized. Rename or delete files that are no longer relevant. "
    "Do not create new files unless necessary.\n\n"
    "IMPORTANT: The memory tool ONLY works with paths under /memories/. "
    "Do NOT use the memory tool to access the repository or workdir. "
    "Use shell commands (ls, cat, etc.) for filesystem access."
)


class AnthropicMemoryTool(MemoryTool, _SDKMemoryTool):
    """
    Anthropic's built-in memory tool, backed by MemoryTool's file logic.

    Inherits file-operation logic from ``MemoryTool`` (plain Python class) and
    the SDK's native dispatch interface from ``BetaAbstractMemoryTool``.

    The SDK command-handler overrides delegate to the inherited private methods
    (``_view``, ``_create``, etc.), translating the SDK ``Command`` objects to
    the ``args: list`` format that those methods expect, and converting the
    returned ``CmdReturn`` to the string that the SDK API requires.

    Parameters
    ----------
    memory_dir : str | Path | None
        Root directory for memory files.  Defaults to ``~/.microbots/memory``.
    usage_instructions : str | None
        Custom instructions appended to the system prompt for the LLM.
        Defaults to ``DEFAULT_MEMORY_INSTRUCTIONS``.
    """

    def __init__(
        self,
        memory_dir: str | Path | None = None,
        usage_instructions: str | None = None,
    ) -> None:
        MemoryTool.__init__(
            self,
            memory_dir=str(memory_dir) if memory_dir else None,
            usage_instructions_to_llm=(
                usage_instructions
                if usage_instructions is not None
                else DEFAULT_MEMORY_INSTRUCTIONS
            ),
        )
        _SDKMemoryTool.__init__(self)  # type: ignore[call-arg]

    # ---------------------------------------------------------------------- #
    # ToolAbstract overrides
    # ---------------------------------------------------------------------- #

    def is_model_supported(self, model_name: str) -> bool:
        """Only Anthropic (Claude) models support the native memory tool."""
        return "claude" in model_name.lower()

    def is_invoked(self, command: str) -> bool:
        """Return True when the command is a serialized native_tool_calls JSON
        containing a call to the ``memory`` tool."""
        try:
            data = json.loads(command)
            if "native_tool_calls" in data:
                return any(tc["name"] == "memory" for tc in data["native_tool_calls"])
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return False

    def invoke(self, command: str, parent_bot) -> CmdReturn:
        """Execute all memory tool calls in the serialized native_tool_calls batch."""
        data = json.loads(command)
        results = []
        for tc in data["native_tool_calls"]:
            if tc["name"] != "memory":
                continue
            try:
                result = self.call(tc["input"])
                logger.info(
                    "\U0001f9e0 Native tool 'memory' executed. Result (first 200 chars): %s",
                    str(result)[:200],
                )
                results.append(str(result))
            except Exception as exc:
                logger.error("Native tool 'memory' raised: %s", exc)
                results.append(f"Error executing tool 'memory': {exc}")
        combined = "\n".join(results)
        return CmdReturn(stdout=combined, stderr="", return_code=0)

    def clear_all(self) -> None:
        """Delete all memory files (useful for testing or resetting state)."""
        self._clear()
        logger.info("🧠 AnthropicMemoryTool: memory cleared at %s", self._memory_dir)

    # ---------------------------------------------------------------------- #
    # BetaAbstractMemoryTool overrides — delegate to MemoryTool private methods
    # ---------------------------------------------------------------------- #

    @override
    def clear_all_memory(self) -> str:
        self.clear_all()
        return "All memory cleared"

    @override
    def view(self, command: BetaMemoryTool20250818ViewCommand) -> str:
        args = [command.path]
        if command.view_range:
            args += ["--start", str(command.view_range[0]), "--end", str(command.view_range[1])]
        result = self._view(args)
        if result.return_code != 0:
            raise RuntimeError(result.stderr)
        return result.stdout

    @override
    def create(self, command: BetaMemoryTool20250818CreateCommand) -> str:
        result = self._create([command.path, command.file_text])
        if result.return_code != 0:
            raise RuntimeError(result.stderr)
        return f"File created successfully at {command.path}"

    @override
    def str_replace(self, command: BetaMemoryTool20250818StrReplaceCommand) -> str:
        result = self._str_replace([command.path, "--old", command.old_str, "--new", command.new_str])
        if result.return_code != 0:
            raise RuntimeError(result.stderr)
        return f"File {command.path} has been edited"

    @override
    def insert(self, command: BetaMemoryTool20250818InsertCommand) -> str:
        result = self._insert([
            command.path,
            "--line", str(command.insert_line),
            "--text", command.insert_text,
        ])
        if result.return_code != 0:
            raise RuntimeError(result.stderr)
        return f"Text inserted at line {command.insert_line} in {command.path}"

    @override
    def delete(self, command: BetaMemoryTool20250818DeleteCommand) -> str:
        result = self._delete([command.path])
        if result.return_code != 0:
            raise RuntimeError(result.stderr)
        return result.stdout

    @override
    def rename(self, command: BetaMemoryTool20250818RenameCommand) -> str:
        result = self._rename([command.old_path, command.new_path])
        if result.return_code != 0:
            raise RuntimeError(result.stderr)
        return result.stdout
