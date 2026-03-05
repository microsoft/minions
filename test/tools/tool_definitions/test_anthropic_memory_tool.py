"""
Unit tests for AnthropicMemoryTool.

Covers:
  - __init__: memory_dir / usage_instructions forwarding and defaults
  - is_model_supported
  - is_invoked
  - clear_all / clear_all_memory (SDK override)
  - SDK overrides: view, create, str_replace, insert, delete, rename
    (happy-path + RuntimeError on failure)
"""
import logging
import pytest

from anthropic.types.beta import (
    BetaMemoryTool20250818CreateCommand,
    BetaMemoryTool20250818DeleteCommand,
    BetaMemoryTool20250818InsertCommand,
    BetaMemoryTool20250818RenameCommand,
    BetaMemoryTool20250818StrReplaceCommand,
    BetaMemoryTool20250818ViewCommand,
)

from microbots.tools.tool_definitions.anthropic_memory_tool import (
    DEFAULT_MEMORY_INSTRUCTIONS,
    AnthropicMemoryTool,
)
from microbots.tools.tool_definitions.memory_tool import MemoryTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tool(tmp_path) -> AnthropicMemoryTool:
    return AnthropicMemoryTool(memory_dir=str(tmp_path / "memory"))


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAnthropicMemoryToolInit:

    def test_is_subclass_of_memory_tool(self, tmp_path):
        assert isinstance(make_tool(tmp_path), MemoryTool)

    def test_memory_dir_is_forwarded(self, tmp_path):
        mem_dir = str(tmp_path / "my_memory")
        tool = AnthropicMemoryTool(memory_dir=mem_dir)
        assert str(tool._memory_dir) == mem_dir

    def test_memory_dir_is_created_on_init(self, tmp_path):
        mem_dir = tmp_path / "new_memory"
        assert not mem_dir.exists()
        AnthropicMemoryTool(memory_dir=str(mem_dir))
        assert mem_dir.exists()

    def test_default_memory_dir_under_home(self, monkeypatch, tmp_path):
        from pathlib import Path
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        tool = AnthropicMemoryTool()
        assert tool._memory_dir == tmp_path / ".microbots" / "memory"

    def test_custom_usage_instructions_are_stored(self, tmp_path):
        custom = "custom instructions"
        tool = AnthropicMemoryTool(
            memory_dir=str(tmp_path / "memory"),
            usage_instructions=custom,
        )
        assert tool.usage_instructions_to_llm == custom

    def test_default_usage_instructions_are_applied_when_none(self, tmp_path):
        tool = make_tool(tmp_path)
        assert tool.usage_instructions_to_llm == DEFAULT_MEMORY_INSTRUCTIONS


# ---------------------------------------------------------------------------
# is_model_supported
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAnthropicMemoryToolIsModelSupported:

    def test_returns_true_for_claude_models(self, tmp_path):
        tool = make_tool(tmp_path)
        for model in ("claude-3-sonnet", "claude-3-5-haiku", "Claude-Opus-4"):
            assert tool.is_model_supported(model) is True

    def test_returns_false_for_non_claude_models(self, tmp_path):
        tool = make_tool(tmp_path)
        for model in ("gpt-4", "ollama/llama3", "azure-openai/gpt-5", ""):
            assert tool.is_model_supported(model) is False


# ---------------------------------------------------------------------------
# is_invoked
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAnthropicMemoryToolIsInvoked:

    def test_always_returns_false(self, tmp_path):
        tool = make_tool(tmp_path)
        for cmd in ("memory view /memories", "memory clear", "anything", ""):
            assert tool.is_invoked(cmd) is False


# ---------------------------------------------------------------------------
# clear_all / clear_all_memory
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAnthropicMemoryToolClearAll:

    def test_clear_all_removes_all_files(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "a.md").write_text("a")
        (tool._memory_dir / "b.md").write_text("b")

        tool.clear_all()

        assert list(tool._memory_dir.iterdir()) == []

    def test_clear_all_leaves_memory_dir_intact(self, tmp_path):
        tool = make_tool(tmp_path)
        tool.clear_all()
        assert tool._memory_dir.exists()

    def test_clear_all_logs_info(self, tmp_path, caplog):
        tool = make_tool(tmp_path)
        with caplog.at_level(logging.INFO):
            tool.clear_all()
        assert "AnthropicMemoryTool" in caplog.text
        assert "cleared" in caplog.text

    def test_clear_all_memory_returns_string(self, tmp_path):
        tool = make_tool(tmp_path)
        result = tool.clear_all_memory()
        assert result == "All memory cleared"

    def test_clear_all_memory_removes_files(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("data")

        tool.clear_all_memory()

        assert list(tool._memory_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# view (SDK override)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAnthropicMemoryToolView:

    def test_view_returns_file_contents(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "notes.md").write_text("hello\nworld\n")

        cmd = BetaMemoryTool20250818ViewCommand(
            command="view", path="/memories/notes.md", view_range=None
        )
        result = tool.view(cmd)

        assert "hello" in result
        assert "world" in result

    def test_view_with_view_range(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("a\nb\nc\nd\ne\n")

        cmd = BetaMemoryTool20250818ViewCommand(
            command="view", path="/memories/f.md", view_range=[2, 4]
        )
        result = tool.view(cmd)

        assert "b" in result
        assert "d" in result
        assert "a" not in result
        assert "e" not in result

    def test_view_raises_runtime_error_on_failure(self, tmp_path):
        tool = make_tool(tmp_path)
        cmd = BetaMemoryTool20250818ViewCommand(
            command="view", path="/memories/nonexistent.md", view_range=None
        )
        with pytest.raises(RuntimeError):
            tool.view(cmd)


# ---------------------------------------------------------------------------
# create (SDK override)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAnthropicMemoryToolCreate:

    def test_create_writes_file(self, tmp_path):
        tool = make_tool(tmp_path)
        cmd = BetaMemoryTool20250818CreateCommand(
            command="create", path="/memories/new.md", file_text="hello world"
        )
        result = tool.create(cmd)

        assert "new.md" in result
        assert (tool._memory_dir / "new.md").read_text() == "hello world"

    def test_create_raises_runtime_error_on_failure(self, tmp_path):
        tool = make_tool(tmp_path)
        # Path traversal should cause _create to fail via _resolve
        cmd = BetaMemoryTool20250818CreateCommand(
            command="create", path="/memories/../../etc/evil.md", file_text="x"
        )
        with pytest.raises((RuntimeError, ValueError)):
            tool.create(cmd)


# ---------------------------------------------------------------------------
# str_replace (SDK override)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAnthropicMemoryToolStrReplace:

    def test_str_replace_edits_file(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("hello world")

        cmd = BetaMemoryTool20250818StrReplaceCommand(
            command="str_replace",
            path="/memories/f.md",
            old_str="hello",
            new_str="goodbye",
        )
        result = tool.str_replace(cmd)

        assert "f.md" in result
        assert (tool._memory_dir / "f.md").read_text() == "goodbye world"

    def test_str_replace_raises_runtime_error_on_failure(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("hello world")
        cmd = BetaMemoryTool20250818StrReplaceCommand(
            command="str_replace",
            path="/memories/f.md",
            old_str="not present",
            new_str="x",
        )
        with pytest.raises(RuntimeError):
            tool.str_replace(cmd)


# ---------------------------------------------------------------------------
# insert (SDK override)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAnthropicMemoryToolInsert:

    def test_insert_prepends_line(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("line1\nline2\n")

        cmd = BetaMemoryTool20250818InsertCommand(
            command="insert",
            path="/memories/f.md",
            insert_line=0,
            insert_text="prepended",
        )
        result = tool.insert(cmd)

        assert "0" in result or "prepended" in result or "f.md" in result
        lines = (tool._memory_dir / "f.md").read_text().splitlines()
        assert lines[0] == "prepended"

    def test_insert_raises_runtime_error_on_failure(self, tmp_path):
        tool = make_tool(tmp_path)
        cmd = BetaMemoryTool20250818InsertCommand(
            command="insert",
            path="/memories/missing.md",
            insert_line=0,
            insert_text="x",
        )
        with pytest.raises(RuntimeError):
            tool.insert(cmd)


# ---------------------------------------------------------------------------
# delete (SDK override)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAnthropicMemoryToolDelete:

    def test_delete_removes_file(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "f.md").write_text("data")

        cmd = BetaMemoryTool20250818DeleteCommand(
            command="delete", path="/memories/f.md"
        )
        tool.delete(cmd)

        assert not (tool._memory_dir / "f.md").exists()

    def test_delete_raises_runtime_error_on_failure(self, tmp_path):
        tool = make_tool(tmp_path)
        cmd = BetaMemoryTool20250818DeleteCommand(
            command="delete", path="/memories/nonexistent.md"
        )
        with pytest.raises(RuntimeError):
            tool.delete(cmd)


# ---------------------------------------------------------------------------
# rename (SDK override)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAnthropicMemoryToolRename:

    def test_rename_moves_file(self, tmp_path):
        tool = make_tool(tmp_path)
        (tool._memory_dir / "old.md").write_text("content")

        cmd = BetaMemoryTool20250818RenameCommand(
            command="rename",
            old_path="/memories/old.md",
            new_path="/memories/new.md",
        )
        tool.rename(cmd)

        assert not (tool._memory_dir / "old.md").exists()
        assert (tool._memory_dir / "new.md").read_text() == "content"

    def test_rename_raises_runtime_error_on_failure(self, tmp_path):
        tool = make_tool(tmp_path)
        cmd = BetaMemoryTool20250818RenameCommand(
            command="rename",
            old_path="/memories/missing.md",
            new_path="/memories/new.md",
        )
        with pytest.raises(RuntimeError):
            tool.rename(cmd)
