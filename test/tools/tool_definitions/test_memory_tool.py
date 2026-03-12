"""
Unit tests for MemoryTool — file-backed memory store.

All tests use pytest's tmp_path fixture so they are isolated from the
user's real ~/.microbots/memory directory.
"""
import pytest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

from microbots.tools.tool_definitions.memory_tool import MemoryTool
from microbots.environment.Environment import CmdReturn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tool(tmp_path: Path) -> MemoryTool:
    """Return a MemoryTool whose memory_dir lives under tmp_path."""
    return MemoryTool(memory_dir=str(tmp_path / "memory"))


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMemoryToolInit:

    def test_memory_dir_is_created_on_init(self, tmp_path):
        mem_dir = tmp_path / "memory"
        assert not mem_dir.exists()

        make_tool(tmp_path)

        assert mem_dir.exists()
        assert mem_dir.is_dir()

    def test_default_memory_dir_under_home(self, monkeypatch, tmp_path):
        """When no memory_dir is given it falls back to ~/.microbots/memory."""
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        tool = MemoryTool()
        assert tool._memory_dir == tmp_path / ".microbots" / "memory"


# ---------------------------------------------------------------------------
# is_invoked
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMemoryToolIsInvoked:

    def test_returns_true_for_memory_commands(self, tmp_path):
        tool = make_tool(tmp_path)
        assert tool.is_invoked("memory view /memories") is True
        assert tool.is_invoked("memory create /memories/f.md hello") is True

    def test_returns_true_for_bare_memory(self, tmp_path):
        tool = make_tool(tmp_path)
        assert tool.is_invoked("memory") is True
        assert tool.is_invoked("memory\n") is True
        assert tool.is_invoked("  memory  ") is True

    def test_returns_false_for_other_commands(self, tmp_path):
        tool = make_tool(tmp_path)
        assert tool.is_invoked("ls -la") is False
        assert tool.is_invoked("cat file.txt") is False
        assert tool.is_invoked("") is False

    def test_strips_leading_whitespace(self, tmp_path):
        tool = make_tool(tmp_path)
        assert tool.is_invoked("  memory view /memories") is True


# ---------------------------------------------------------------------------
# Path resolution (_resolve)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMemoryToolResolve:

    def test_resolve_memories_root(self, tmp_path):
        tool = make_tool(tmp_path)
        assert tool._resolve("/memories") == tool._memory_dir.resolve()

    def test_resolve_memories_subpath(self, tmp_path):
        tool = make_tool(tmp_path)
        resolved = tool._resolve("/memories/notes.md")
        assert resolved == (tool._memory_dir / "notes.md").resolve()

    def test_resolve_rejects_path_traversal(self, tmp_path):
        tool = make_tool(tmp_path)
        with pytest.raises(ValueError, match="Path traversal"):
            tool._resolve("/memories/../../etc/passwd")

    def test_resolve_rejects_symlink_escaping_memory_dir(self, tmp_path):
        """A symlink inside memory_dir that resolves outside must be rejected
        by the startswith guard (not the '..' check)."""
        tool = make_tool(tmp_path)
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("sensitive")
        # Create a symlink inside the memory dir pointing outside
        link = tool._memory_dir / "escape"
        link.symlink_to(outside)
        with pytest.raises(ValueError, match="Path traversal"):
            tool._resolve("/memories/escape/secret.txt")

    def test_resolve_rejects_non_memory_paths(self, tmp_path):
        tool = make_tool(tmp_path)
        for bad in ("/workdir/file", "/home/user/file", "/tmp/file"):
            with pytest.raises(ValueError):
                tool._resolve(bad)

    def test_resolve_bare_relative_path_treated_as_relative_to_memory_dir(self, tmp_path):
        """The else branch: a path without a /memories/ prefix is resolved
        relative to memory_dir."""
        tool = make_tool(tmp_path)
        resolved = tool._resolve("notes.md")
        assert resolved == (tool._memory_dir / "notes.md").resolve()

    def test_resolve_bare_relative_subdir_path(self, tmp_path):
        """A bare relative path with subdirectory components is also resolved
        relative to memory_dir (else branch)."""
        tool = make_tool(tmp_path)
        resolved = tool._resolve("sub/dir/file.md")
        assert resolved == (tool._memory_dir / "sub" / "dir" / "file.md").resolve()


# ---------------------------------------------------------------------------
# _view
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMemoryToolView:

    def test_view_directory_lists_contents(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "notes.md").write_text("hello")
        (tool._memory_dir / "sub").mkdir()

        result = tool._view(Namespace(path="/memories", start=None, end=None))

        assert result.return_code == 0
        assert "notes.md" in result.stdout
        assert "sub/" in result.stdout

    def test_view_file_returns_numbered_lines(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("line1\nline2\nline3\n")

        result = tool._view(Namespace(path="/memories/f.md", start=None, end=None))

        assert result.return_code == 0
        assert "1:" in result.stdout
        assert "line1" in result.stdout
        assert "3:" in result.stdout

    def test_view_file_with_line_range(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("a\nb\nc\nd\ne\n")

        result = tool._view(Namespace(path="/memories/f.md", start=2, end=4))

        assert result.return_code == 0
        assert "b" in result.stdout
        assert "d" in result.stdout
        assert "a" not in result.stdout
        assert "e" not in result.stdout

    def test_view_nonexistent_path_returns_error(self, tmp_path):
        tool = make_tool(tmp_path)

        result = tool._view(Namespace(path="/memories/nonexistent.md", start=None, end=None))

        assert result.return_code != 0

    def test_view_no_args_returns_error(self, tmp_path):
        tool = make_tool(tmp_path)
        result = tool.invoke("memory view", parent_bot=Mock())
        assert result.return_code != 0

    def test_view_unknown_flag_returns_error(self, tmp_path):
        """Argparse rejects unrecognised flags."""
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("hello\n")
        result = tool.invoke("memory view /memories/f.md --bogus value", parent_bot=Mock())
        assert result.return_code != 0


# ---------------------------------------------------------------------------
# _create
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMemoryToolCreate:

    def test_create_writes_file(self, tmp_path):
        tool = make_tool(tmp_path)

        result = tool._create(Namespace(path="/memories/notes.md", content=["hello world"]))

        assert result.return_code == 0
        assert (tool._memory_dir / "notes.md").read_text() == "hello world"

    def test_create_overwrites_existing_file(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("old content")

        result = tool._create(Namespace(path="/memories/f.md", content=["new content"]))

        assert result.return_code == 0
        assert (tool._memory_dir / "f.md").read_text() == "new content"

    def test_create_creates_parent_directories(self, tmp_path):
        tool = make_tool(tmp_path)

        result = tool._create(Namespace(path="/memories/sub/dir/f.md", content=["content"]))

        assert result.return_code == 0
        assert (tool._memory_dir / "sub" / "dir" / "f.md").exists()

    def test_create_missing_args_returns_error(self, tmp_path):
        tool = make_tool(tmp_path)
        result = tool.invoke("memory create /memories/f.md", parent_bot=Mock())  # missing content
        assert result.return_code != 0


# ---------------------------------------------------------------------------
# _str_replace
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMemoryToolStrReplace:

    def test_str_replace_replaces_unique_text(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("hello world")

        result = tool._str_replace(Namespace(path="/memories/f.md", old="hello", new="goodbye"))

        assert result.return_code == 0
        assert (tool._memory_dir / "f.md").read_text() == "goodbye world"

    def test_str_replace_fails_when_text_not_found(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("hello world")

        result = tool._str_replace(Namespace(path="/memories/f.md", old="nothere", new="x"))

        assert result.return_code != 0
        assert "not found" in result.stderr.lower()

    def test_str_replace_fails_when_text_not_unique(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("hello hello")

        result = tool._str_replace(Namespace(path="/memories/f.md", old="hello", new="bye"))

        assert result.return_code != 0
        assert "2" in result.stderr  # appears N times

    def test_str_replace_missing_flags_returns_error(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("text")
        result = tool.invoke("memory str_replace /memories/f.md", parent_bot=Mock())
        assert result.return_code != 0

    def test_str_replace_empty_args_returns_usage_error(self, tmp_path):
        """Calling str_replace with no args returns an error via argparse."""
        tool = make_tool(tmp_path)
        result = tool.invoke("memory str_replace", parent_bot=Mock())
        assert result.return_code == 1

    def test_str_replace_nonexistent_file_returns_error(self, tmp_path):
        tool = make_tool(tmp_path)
        result = tool._str_replace(Namespace(path="/memories/missing.md", old="a", new="b"))
        assert result.return_code != 0

    def test_str_replace_unknown_flag_returns_error(self, tmp_path):
        """Argparse rejects unrecognised flags."""
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("hello world")
        result = tool.invoke(
            'memory str_replace /memories/f.md --bogus ignored --old "hello" --new "goodbye"',
            parent_bot=Mock(),
        )
        assert result.return_code != 0


# ---------------------------------------------------------------------------
# _insert
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMemoryToolInsert:

    def test_insert_prepends_at_line_zero(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("line1\nline2\n")

        result = tool._insert(Namespace(path="/memories/f.md", line=0, text="prepended"))

        assert result.return_code == 0
        lines = (tool._memory_dir / "f.md").read_text().splitlines()
        assert lines[0] == "prepended"
        assert lines[1] == "line1"

    def test_insert_at_end_of_file(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("line1\nline2\n")

        result = tool._insert(Namespace(path="/memories/f.md", line=2, text="appended"))

        assert result.return_code == 0
        lines = (tool._memory_dir / "f.md").read_text().splitlines()
        assert lines[-1] == "appended"

    def test_insert_invalid_line_number_returns_error(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("line1\n")

        result = tool._insert(Namespace(path="/memories/f.md", line=99, text="x"))

        assert result.return_code != 0

    def test_insert_nonexistent_file_returns_error(self, tmp_path):
        tool = make_tool(tmp_path)
        result = tool._insert(Namespace(path="/memories/missing.md", line=0, text="x"))
        assert result.return_code != 0

    def test_insert_missing_flags_returns_error(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("line1\n")
        result = tool.invoke("memory insert /memories/f.md", parent_bot=Mock())
        assert result.return_code != 0

    def test_insert_empty_args_returns_usage_error(self, tmp_path):
        """Calling insert with no args returns an error via argparse."""
        tool = make_tool(tmp_path)
        result = tool.invoke("memory insert", parent_bot=Mock())
        assert result.return_code == 1

    def test_insert_unknown_flag_returns_error(self, tmp_path):
        """Argparse rejects unrecognised flags."""
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("line1\nline2\n")
        result = tool.invoke(
            'memory insert /memories/f.md --bogus ignored --line 0 --text "prepended"',
            parent_bot=Mock(),
        )
        assert result.return_code != 0


# ---------------------------------------------------------------------------
# _delete
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMemoryToolDelete:

    def test_delete_removes_file(self, tmp_path):
        tool = make_tool(tmp_path)
        f = tool._memory_dir / "f.md"
        f.write_text("data")

        result = tool._delete(Namespace(path="/memories/f.md"))

        assert result.return_code == 0
        assert not f.exists()

    def test_delete_removes_directory(self, tmp_path):
        tool = make_tool(tmp_path)
        sub = tool._memory_dir / "sub"
        sub.mkdir()
        (sub / "f.md").write_text("data")

        result = tool._delete(Namespace(path="/memories/sub"))

        assert result.return_code == 0
        assert not sub.exists()

    def test_delete_prevents_root_deletion(self, tmp_path):
        tool = make_tool(tmp_path)
        for path in ("/memories", "memories", "/memories/"):
            result = tool._delete(Namespace(path=path))
            assert result.return_code != 0

    def test_delete_nonexistent_path_raises(self, tmp_path):
        tool = make_tool(tmp_path)
        result = tool._delete(Namespace(path="/memories/nonexistent.md"))
        assert result.return_code != 0

    def test_delete_no_args_returns_error(self, tmp_path):
        tool = make_tool(tmp_path)
        result = tool.invoke("memory delete", parent_bot=Mock())
        assert result.return_code != 0


# ---------------------------------------------------------------------------
# _rename
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMemoryToolRename:

    def test_rename_moves_file(self, tmp_path):
        tool = make_tool(tmp_path)
        src = tool._memory_dir / "old.md"
        src.write_text("content")

        result = tool._rename(Namespace(old_path="/memories/old.md", new_path="/memories/new.md"))

        assert result.return_code == 0
        assert not src.exists()
        assert (tool._memory_dir / "new.md").read_text() == "content"

    def test_rename_nonexistent_source_returns_error(self, tmp_path):
        tool = make_tool(tmp_path)
        result = tool._rename(Namespace(old_path="/memories/missing.md", new_path="/memories/new.md"))
        assert result.return_code != 0

    def test_rename_fails_if_destination_exists(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "a.md").write_text("a")
        (tool._memory_dir / "b.md").write_text("b")

        result = tool._rename(Namespace(old_path="/memories/a.md", new_path="/memories/b.md"))

        assert result.return_code != 0

    def test_rename_missing_args_returns_error(self, tmp_path):
        tool = make_tool(tmp_path)
        result = tool.invoke("memory rename /memories/a.md", parent_bot=Mock())
        assert result.return_code != 0


# ---------------------------------------------------------------------------
# _clear
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMemoryToolClear:

    def test_clear_removes_all_files(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "a.md").write_text("a")
        (tool._memory_dir / "b.md").write_text("b")

        result = tool._clear()

        assert result.return_code == 0
        assert list(tool._memory_dir.iterdir()) == []

    def test_clear_leaves_memory_dir_intact(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("data")

        tool._clear()

        assert tool._memory_dir.exists()


# ---------------------------------------------------------------------------
# is_model_supported
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMemoryToolIsModelSupported:

    def test_returns_true_for_any_model(self, tmp_path):
        tool = make_tool(tmp_path)
        for model in ("gpt-4", "claude-3-sonnet", "ollama/llama3", ""):
            assert tool.is_model_supported(model) is True


# ---------------------------------------------------------------------------
# invoke — full command dispatch
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMemoryToolInvoke:

    def test_invoke_view_subcommand(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("hello")

        result = tool.invoke("memory view /memories/f.md", parent_bot=Mock())

        assert result.return_code == 0
        assert "hello" in result.stdout

    def test_invoke_create_subcommand(self, tmp_path):
        tool = make_tool(tmp_path)

        result = tool.invoke('memory create /memories/n.md "some content"', parent_bot=Mock())

        assert result.return_code == 0
        assert (tool._memory_dir / "n.md").read_text() == "some content"

    def test_invoke_clear_subcommand(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("data")

        result = tool.invoke("memory clear", parent_bot=Mock())

        assert result.return_code == 0
        assert list(tool._memory_dir.iterdir()) == []

    def test_invoke_unknown_subcommand_returns_error(self, tmp_path):
        tool = make_tool(tmp_path)
        result = tool.invoke("memory frobnicate /memories/f.md", parent_bot=Mock())
        assert result.return_code != 0
        assert "invalid choice" in result.stderr

    def test_invoke_too_few_tokens_returns_error(self, tmp_path):
        tool = make_tool(tmp_path)
        result = tool.invoke("memory", parent_bot=Mock())
        assert result.return_code != 0

    def test_invoke_handles_bad_quoting_gracefully(self, tmp_path):
        tool = make_tool(tmp_path)
        result = tool.invoke('memory create /memories/f.md "unclosed', parent_bot=Mock())
        assert result.return_code != 0
        assert "Parse error" in result.stderr

    def test_invoke_str_replace_subcommand(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("hello world")

        result = tool.invoke(
            'memory str_replace /memories/f.md --old "hello" --new "goodbye"',
            parent_bot=Mock(),
        )

        assert result.return_code == 0
        assert (tool._memory_dir / "f.md").read_text() == "goodbye world"

    def test_invoke_insert_subcommand(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("line1\nline2\n")

        result = tool.invoke(
            'memory insert /memories/f.md --line 0 --text "prepended"',
            parent_bot=Mock(),
        )

        assert result.return_code == 0
        lines = (tool._memory_dir / "f.md").read_text().splitlines()
        assert lines[0] == "prepended"

    def test_invoke_delete_subcommand(self, tmp_path):
        tool = make_tool(tmp_path)
        f = tool._memory_dir / "f.md"
        f.write_text("data")

        result = tool.invoke("memory delete /memories/f.md", parent_bot=Mock())

        assert result.return_code == 0
        assert not f.exists()

    def test_invoke_rename_subcommand(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "old.md").write_text("content")

        result = tool.invoke(
            "memory rename /memories/old.md /memories/new.md",
            parent_bot=Mock(),
        )

        assert result.return_code == 0
        assert (tool._memory_dir / "new.md").read_text() == "content"
        assert not (tool._memory_dir / "old.md").exists()

    def test_invoke_exception_returned_as_error_cmdreturn(self, tmp_path):
        """except (ValueError, FileNotFoundError, RuntimeError) block:
        a path-traversal path causes _resolve() to raise ValueError inside a
        subcommand handler, which is caught and returned as CmdReturn(return_code=1)."""
        tool = make_tool(tmp_path)

        # Path traversal triggers ValueError inside _view → caught by except block
        result = tool.invoke(
            "memory view /memories/../../etc/passwd",
            parent_bot=Mock(),
        )

        assert result.return_code == 1
        assert result.stdout == ""
        assert "traversal" in result.stderr.lower() or result.stderr != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
