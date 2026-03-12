import logging
import os
import shlex
import shutil
from pathlib import Path
from typing import Optional

from pydantic.dataclasses import dataclass, Field

from microbots.environment.Environment import CmdReturn
from microbots.tools.external_tool import ExternalTool

logger = logging.getLogger(" 🧠 MemoryTool")

INSTRUCTIONS_TO_LLM = """
Use this tool to persist information to files across steps — same interface as
the Anthropic memory tool.  All paths must be under /memories/.

MEMORY PROTOCOL:
1. ALWAYS run `memory view /memories` BEFORE doing anything else to check for
   earlier progress.
2. Record status, findings and intermediate results as you go.
3. Before completing a task, save your final results to memory.
4. Keep the memory folder organised — rename or delete stale files.

## Commands

View a file or list a directory:
  memory view <path>
  memory view <path> --start <line> --end <line>

Create a file:
  memory create <path> <content>

Replace a unique string in a file:
  memory str_replace <path> --old "<old_text>" --new "<new_text>"

Insert a line into a file (0 = prepend):
  memory insert <path> --line <line_number> --text "<text>"

Delete a file or directory:
  memory delete <path>

Rename / move a file:
  memory rename <old_path> <new_path>

Clear all memory:
  memory clear

## Examples

  memory view /memories
  memory create /memories/progress.md "## Progress\\n- Found bug in src/foo.py line 42"
  memory str_replace /memories/progress.md --old "line 42" --new "line 45"
  memory insert /memories/progress.md --line 0 --text "# Task Notes"
  memory view /memories/progress.md --start 1 --end 10
  memory delete /memories/old_notes.md
  memory rename /memories/draft.md /memories/final.md

## Notes
- Paths must start with /memories/.
- memory create overwrites if the file already exists.
- memory str_replace requires the old text to appear exactly once.
- In memory view, use --end -1 to read through the end of the file.
"""


@dataclass
class MemoryTool(ExternalTool):
    """
    File-backed memory tool that dispatches through the text command loop and
    works consistently across providers.

    Subclass of ``ExternalTool`` — all command lists are empty so
    ``install_tool``, ``setup_tool``, ``verify_tool_installation``, and
    ``uninstall_tool`` are all effective no-ops inherited from ``ExternalTool``.

    All files are stored under ``memory_dir`` on the host (default
    ``~/.microbots/memory``).  The LLM uses paths like ``/memories/notes.md``
    which are resolved relative to ``memory_dir``.

    Supported subcommands
    ---------------------
    memory view <path> [--start N] [--end N]
    memory create <path> <content>
    memory str_replace <path> --old <text> --new <text>
    memory insert <path> --line N --text <text>
    memory delete <path>
    memory rename <old> <new>
    memory clear
    """

    name: str = Field(default="memory")
    description: str = Field(
        default="File-backed memory store — view, create, edit, delete files under /memories/."
    )
    usage_instructions_to_llm: str = Field(default=INSTRUCTIONS_TO_LLM)
    memory_dir: Optional[str] = Field(default=None)

    def __post_init__(self):
        base = Path(self.memory_dir) if self.memory_dir else Path.home() / ".microbots" / "memory"
        self._memory_dir = base
        self._memory_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------- #
    # Path helpers
    # ---------------------------------------------------------------------- #

    def _resolve(self, path: str) -> Path:
        """Resolve a /memories/… path to an absolute host path."""
        stripped = path.lstrip("/")

        # Reject any path containing '..' components before resolving
        if ".." in Path(stripped).parts:
            raise ValueError(f"Path traversal not allowed: {path!r}")

        if path.startswith("/") and stripped != "memories" and not stripped.startswith("memories/"):
            raise ValueError(
                f"Invalid memory path: {path!r}. Use paths under /memories/."
            )

        if stripped == "memories":
            rel = ""
        elif stripped.startswith("memories/"):
            rel = stripped[len("memories/"):]
        else:
            rel = stripped  # treat as relative to memory_dir

        resolved = (self._memory_dir / rel).resolve() if rel else self._memory_dir.resolve()
        # Use trailing separator to prevent prefix confusion with sibling dirs
        memory_root = str(self._memory_dir.resolve())
        if resolved != self._memory_dir.resolve() and not str(resolved).startswith(memory_root + os.sep):
            raise ValueError(f"Path traversal not allowed: {path!r}")
        return resolved

    # ---------------------------------------------------------------------- #
    # ToolAbstract interface
    # ---------------------------------------------------------------------- #

    def is_invoked(self, command: str) -> bool:
        return command.strip().startswith("memory ")

    def invoke(self, command: str, parent_bot) -> CmdReturn:
        try:
            tokens = shlex.split(command)
        except ValueError as exc:
            return CmdReturn(stdout="", stderr=f"Parse error: {exc}", return_code=1)

        if len(tokens) < 2:
            return CmdReturn(stdout="", stderr="Usage: memory <subcommand> ...", return_code=1)

        sub = tokens[1]
        args = tokens[2:]

        try:
            if sub == "view":
                return self._view(args)
            elif sub == "create":
                return self._create(args)
            elif sub == "str_replace":
                return self._str_replace(args)
            elif sub == "insert":
                return self._insert(args)
            elif sub == "delete":
                return self._delete(args)
            elif sub == "rename":
                return self._rename(args)
            elif sub == "clear":
                return self._clear()
            else:
                return CmdReturn(stdout="", stderr=f"Unknown subcommand: {sub!r}", return_code=1)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            logger.error("🧠 MemoryTool error: %s", exc)
            return CmdReturn(stdout="", stderr=str(exc), return_code=1)

    # ---------------------------------------------------------------------- #
    # Subcommand handlers
    # ---------------------------------------------------------------------- #

    def _view(self, args: list) -> CmdReturn:
        if not args:
            return CmdReturn(stdout="", stderr="Usage: memory view <path> [--start N] [--end N]", return_code=1)

        path = args[0]
        start_line = None
        end_line = None
        i = 1
        while i < len(args):
            if args[i] == "--start" and i + 1 < len(args):
                start_line = int(args[i + 1]); i += 2
            elif args[i] == "--end" and i + 1 < len(args):
                end_line = int(args[i + 1]); i += 2
            else:
                logger.warning("🧠 MemoryTool view: unknown flag %r (skipped)", args[i])
                i += 1

        resolved = self._resolve(path)
        if not resolved.exists():
            return CmdReturn(stdout="", stderr=f"Path not found: {path!r}", return_code=1)

        if resolved.is_dir():
            items = [
                (f"{item.name}/" if item.is_dir() else item.name)
                for item in sorted(resolved.iterdir())
                if not item.name.startswith(".")
            ]
            result = f"Directory: {path}\n" + "\n".join(f"- {i}" for i in items)
            return CmdReturn(stdout=result, stderr="", return_code=0)

        lines = resolved.read_text(encoding="utf-8").splitlines()
        if start_line is not None or end_line is not None:
            s = max(0, (start_line or 1) - 1)
            e = len(lines) if (end_line is None or end_line == -1) else end_line
            lines = lines[s:e]
            base_num = s + 1
        else:
            base_num = 1
        numbered = "\n".join(f"{i + base_num:4d}: {line}" for i, line in enumerate(lines))
        return CmdReturn(stdout=numbered, stderr="", return_code=0)

    def _create(self, args: list) -> CmdReturn:
        if len(args) < 2:
            return CmdReturn(stdout="", stderr="Usage: memory create <path> <content>", return_code=1)
        path, content = args[0], args[1]
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        logger.info("🧠 Memory file created: %s", path)
        return CmdReturn(stdout=f"File created: {path}", stderr="", return_code=0)

    def _str_replace(self, args: list) -> CmdReturn:
        if not args:
            return CmdReturn(stdout="", stderr="Usage: memory str_replace <path> --old <text> --new <text>", return_code=1)
        path = args[0]
        old_text = new_text = None
        i = 1
        while i < len(args):
            if args[i] == "--old" and i + 1 < len(args):
                old_text = args[i + 1]; i += 2
            elif args[i] == "--new" and i + 1 < len(args):
                new_text = args[i + 1]; i += 2
            else:
                logger.warning("🧠 MemoryTool str_replace: unknown flag %r (skipped)", args[i])
                i += 1
        if old_text is None or new_text is None:
            return CmdReturn(stdout="", stderr="--old and --new are required", return_code=1)
        resolved = self._resolve(path)
        if not resolved.is_file():
            return CmdReturn(stdout="", stderr=f"File not found: {path!r}", return_code=1)
        content = resolved.read_text(encoding="utf-8")
        count = content.count(old_text)
        if count == 0:
            return CmdReturn(stdout="", stderr=f"Text not found in {path!r}", return_code=1)
        if count > 1:
            return CmdReturn(stdout="", stderr=f"Text appears {count} times in {path!r} — must be unique", return_code=1)
        resolved.write_text(content.replace(old_text, new_text, 1), encoding="utf-8")
        return CmdReturn(stdout=f"File {path} edited.", stderr="", return_code=0)

    def _insert(self, args: list) -> CmdReturn:
        if not args:
            return CmdReturn(stdout="", stderr="Usage: memory insert <path> --line N --text <text>", return_code=1)
        path = args[0]
        line_num = text = None
        i = 1
        while i < len(args):
            if args[i] == "--line" and i + 1 < len(args):
                line_num = int(args[i + 1]); i += 2
            elif args[i] == "--text" and i + 1 < len(args):
                text = args[i + 1]; i += 2
            else:
                logger.warning("🧠 MemoryTool insert: unknown flag %r (skipped)", args[i])
                i += 1
        if line_num is None or text is None:
            return CmdReturn(stdout="", stderr="--line and --text are required", return_code=1)
        resolved = self._resolve(path)
        if not resolved.is_file():
            return CmdReturn(stdout="", stderr=f"File not found: {path!r}", return_code=1)
        lines = resolved.read_text(encoding="utf-8").splitlines()
        if line_num < 0 or line_num > len(lines):
            return CmdReturn(stdout="", stderr=f"Invalid line number {line_num}. Must be 0–{len(lines)}.", return_code=1)
        lines.insert(line_num, text.rstrip("\n"))
        resolved.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return CmdReturn(stdout=f"Text inserted at line {line_num} in {path}.", stderr="", return_code=0)

    def _delete(self, args: list) -> CmdReturn:
        if not args:
            return CmdReturn(stdout="", stderr="Usage: memory delete <path>", return_code=1)
        path = args[0]
        if path.rstrip("/") in ("/memories", "memories", ""):
            return CmdReturn(stdout="", stderr="Cannot delete the /memories root directory", return_code=1)
        resolved = self._resolve(path)
        if resolved.is_file():
            resolved.unlink()
            logger.info("🧠 Memory file deleted: %s", path)
            return CmdReturn(stdout=f"Deleted: {path}", stderr="", return_code=0)
        if resolved.is_dir():
            shutil.rmtree(resolved)
            logger.info("🧠 Memory directory deleted: %s", path)
            return CmdReturn(stdout=f"Deleted directory: {path}", stderr="", return_code=0)
        return CmdReturn(stdout="", stderr=f"Path not found: {path!r}", return_code=1)

    def _rename(self, args: list) -> CmdReturn:
        if len(args) < 2:
            return CmdReturn(stdout="", stderr="Usage: memory rename <old_path> <new_path>", return_code=1)
        old_path, new_path = args[0], args[1]
        old_resolved = self._resolve(old_path)
        new_resolved = self._resolve(new_path)
        if not old_resolved.exists():
            return CmdReturn(stdout="", stderr=f"Source not found: {old_path!r}", return_code=1)
        if new_resolved.exists():
            return CmdReturn(stdout="", stderr=f"Destination already exists: {new_path!r}", return_code=1)
        new_resolved.parent.mkdir(parents=True, exist_ok=True)
        old_resolved.rename(new_resolved)
        logger.info("🧠 Memory renamed: %s → %s", old_path, new_path)
        return CmdReturn(stdout=f"Renamed {old_path} to {new_path}.", stderr="", return_code=0)

    def _clear(self) -> CmdReturn:
        if self._memory_dir.exists():
            shutil.rmtree(self._memory_dir)
            self._memory_dir.mkdir(parents=True, exist_ok=True)
        logger.info("🧠 Memory cleared.")
        return CmdReturn(stdout="Memory cleared.", stderr="", return_code=0)
