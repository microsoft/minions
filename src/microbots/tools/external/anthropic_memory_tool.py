"""
AnthropicMemoryTool — wraps Anthropic's memory tool (beta).

The memory tool lets the model persist information across conversations by
reading and writing files in a local memory directory.  When the model invokes
the tool, it sends a command (view, create, str_replace, insert, delete,
rename) and the client executes it against a local filesystem directory.

This implementation extends the SDK's ``BetaAbstractMemoryTool`` for native
integration with ``tool_runner``, and also inherits from ``ExternalTool`` so
the Microbots framework can classify it correctly via ``isinstance`` checks.

Usage:
    from microbots.tools import AnthropicMemoryTool

    memory = AnthropicMemoryTool()
    bot = ReadingBot(..., additional_tools=[memory])

    # Or, at the lower level:
    # llm = AnthropicApi(..., external_tools=[memory])
"""

from __future__ import annotations

import os
import shutil
from logging import getLogger
from pathlib import Path
from typing import List

from typing_extensions import override

from anthropic.lib.tools import BetaAbstractMemoryTool as _SDKMemoryTool
from anthropic.types.beta import (
    BetaMemoryTool20250818ViewCommand,
    BetaMemoryTool20250818CreateCommand,
    BetaMemoryTool20250818DeleteCommand,
    BetaMemoryTool20250818InsertCommand,
    BetaMemoryTool20250818RenameCommand,
    BetaMemoryTool20250818StrReplaceCommand,
)

from microbots.tools.external_tool import ExternalTool

logger = getLogger(__name__)

MEMORY_TOOL_TYPE = "memory_20250818"
MEMORY_BETA_HEADER = "context-management-2025-06-27"


class AnthropicMemoryTool(_SDKMemoryTool, ExternalTool):
    """
    Anthropic's built-in memory tool (beta).

    Extends the SDK's ``BetaAbstractMemoryTool`` for native ``tool_runner``
    integration and ``ExternalTool`` for Microbots framework classification.

    The model sends file-operation commands and this tool executes them
    against a local ``memory_dir`` directory.
    """

    def __init__(self, memory_dir: str | Path | None = None):
        """
        Parameters
        ----------
        memory_dir : str | Path | None
            Root directory for memory files.  Defaults to ``~/.microbots/memory``.
        """
        super().__init__()
        if memory_dir is None:
            memory_dir = Path.home() / ".microbots" / "memory"
        self._memory_dir = Path(memory_dir)
        self._memory_dir.mkdir(parents=True, exist_ok=True)

    @property
    def description(self) -> str:
        return "Persistent memory tool — stores and retrieves information across conversations."

    def get_tool_definition(self) -> dict:
        """Return the tool param dict (delegates to SDK's ``to_dict()``)."""
        return self.to_dict()

    @property
    def beta_header(self) -> str:
        return MEMORY_BETA_HEADER

    @property
    def memory_dir(self) -> Path:
        return self._memory_dir

    def clear_all(self):
        """Delete all memory files (useful for testing)."""
        if self._memory_dir.exists():
            shutil.rmtree(self._memory_dir)
            self._memory_dir.mkdir(parents=True, exist_ok=True)

    @override
    def clear_all_memory(self) -> str:
        """Override the SDK base to provide filesystem clearing."""
        self.clear_all()
        return "All memory cleared"

    def _resolve(self, path: str) -> Path:
        """Resolve a path relative to memory_dir, blocking traversal."""
        # Strip leading slash and "memories/" prefix so the path resolves
        # inside memory_dir (e.g., "/memories/foo.txt" → "foo.txt").
        path = path.lstrip("/")
        if path.startswith("memories/"):
            path = path[len("memories/"):]  # handles both "memories/x" and "memories/"
        elif path == "memories":
            path = ""
        resolved = (self._memory_dir / path).resolve() if path else self._memory_dir.resolve()
        if not str(resolved).startswith(str(self._memory_dir.resolve())):
            raise ValueError(f"Path traversal not allowed: {path}")
        return resolved

    # ---- BetaAbstractMemoryTool typed command handlers -----------------------

    @override
    def view(self, command: BetaMemoryTool20250818ViewCommand) -> str:
        path = command.path
        resolved = self._resolve(path)

        if not resolved.exists():
            raise RuntimeError(f"Path not found: {path}")

        if resolved.is_dir():
            items: List[str] = []
            for item in sorted(resolved.iterdir()):
                if item.name.startswith("."):
                    continue
                items.append(f"{item.name}/" if item.is_dir() else item.name)
            return f"Directory: {path}\n" + "\n".join(f"- {item}" for item in items)

        content = resolved.read_text(encoding="utf-8")
        lines = content.splitlines()
        view_range = command.view_range
        if view_range:
            start_line = max(1, view_range[0]) - 1
            end_line = len(lines) if view_range[1] == -1 else view_range[1]
            lines = lines[start_line:end_line]
            start_num = start_line + 1
        else:
            start_num = 1
        numbered_lines = [f"{i + start_num:4d}: {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered_lines)

    @override
    def create(self, command: BetaMemoryTool20250818CreateCommand) -> str:
        path = command.path
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(command.file_text, encoding="utf-8")
        return f"File created successfully at {path}"

    @override
    def str_replace(self, command: BetaMemoryTool20250818StrReplaceCommand) -> str:
        path = command.path
        resolved = self._resolve(path)

        if not resolved.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        content = resolved.read_text(encoding="utf-8")
        count = content.count(command.old_str)
        if count == 0:
            raise ValueError(f"Text not found in {path}")
        elif count > 1:
            raise ValueError(f"Text appears {count} times in {path}. Must be unique.")

        new_content = content.replace(command.old_str, command.new_str, 1)
        resolved.write_text(new_content, encoding="utf-8")
        return f"File {path} has been edited"

    @override
    def insert(self, command: BetaMemoryTool20250818InsertCommand) -> str:
        path = command.path
        resolved = self._resolve(path)

        if not resolved.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        lines = resolved.read_text(encoding="utf-8").splitlines()
        insert_line = command.insert_line
        insert_text = command.insert_text

        if insert_line < 0 or insert_line > len(lines):
            raise ValueError(f"Invalid insert_line {insert_line}. Must be 0-{len(lines)}")

        lines.insert(insert_line, insert_text.rstrip("\n"))
        resolved.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return f"Text inserted at line {insert_line} in {path}"

    @override
    def delete(self, command: BetaMemoryTool20250818DeleteCommand) -> str:
        path = command.path
        resolved = self._resolve(path)

        if path.rstrip("/") in ("/memories", "memories", ""):
            raise ValueError("Cannot delete the /memories directory itself")

        if resolved.is_file():
            resolved.unlink()
            return f"File deleted: {path}"
        elif resolved.is_dir():
            shutil.rmtree(resolved)
            return f"Directory deleted: {path}"
        else:
            raise FileNotFoundError(f"Path not found: {path}")

    @override
    def rename(self, command: BetaMemoryTool20250818RenameCommand) -> str:
        old_full = self._resolve(command.old_path)
        new_full = self._resolve(command.new_path)

        if not old_full.exists():
            raise FileNotFoundError(f"Source path not found: {command.old_path}")
        if new_full.exists():
            raise ValueError(f"Destination already exists: {command.new_path}")

        new_full.parent.mkdir(parents=True, exist_ok=True)
        old_full.rename(new_full)
        return f"Renamed {command.old_path} to {command.new_path}"
