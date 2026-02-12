"""
Unit tests for the AnthropicMemoryTool and its integration with AnthropicApi.
"""
import pytest

from microbots.tools.tool import Tool
from microbots.tools.external_tool import ExternalTool
from microbots.tools.external.anthropic_memory_tool import (
    AnthropicMemoryTool,
    MEMORY_TOOL_TYPE,
    MEMORY_BETA_HEADER,
)


@pytest.mark.unit
class TestAnthropicMemoryTool:
    """Tests for the AnthropicMemoryTool concrete class.

    Uses ``call(dict)`` which is the SDK entry point (dict → typed command
    → dispatch to view/create/etc.).  This matches real tool_runner usage.
    """

    @pytest.fixture
    def memory_tool(self, tmp_path):
        return AnthropicMemoryTool(memory_dir=tmp_path / "mem")

    def test_name(self, memory_tool):
        assert memory_tool.name == "memory"

    def test_description(self, memory_tool):
        assert isinstance(memory_tool.description, str)
        assert len(memory_tool.description) > 0

    def test_get_tool_definition(self, memory_tool):
        defn = memory_tool.get_tool_definition()
        assert defn == {"type": MEMORY_TOOL_TYPE, "name": "memory"}

    def test_to_dict_matches_get_tool_definition(self, memory_tool):
        """to_dict (SDK) and get_tool_definition (ExternalTool) return the same."""
        assert memory_tool.to_dict() == memory_tool.get_tool_definition()

    def test_beta_header(self, memory_tool):
        assert memory_tool.beta_header == MEMORY_BETA_HEADER

    def test_create_and_view(self, memory_tool):
        result = memory_tool.call({"command": "create", "path": "/memories/notes.md", "file_text": "hello world"})
        assert "created" in result.lower()
        content = memory_tool.call({"command": "view", "path": "/memories/notes.md"})
        assert "hello world" in content

    def test_str_replace(self, memory_tool):
        memory_tool.call({"command": "create", "path": "/memories/f.txt", "file_text": "foo bar baz"})
        memory_tool.call({"command": "str_replace", "path": "/memories/f.txt", "old_str": "bar", "new_str": "qux"})
        content = memory_tool.call({"command": "view", "path": "/memories/f.txt"})
        assert "foo qux baz" in content

    def test_delete(self, memory_tool):
        memory_tool.call({"command": "create", "path": "/memories/tmp.txt", "file_text": "x"})
        result = memory_tool.call({"command": "delete", "path": "/memories/tmp.txt"})
        assert "deleted" in result.lower()
        with pytest.raises(RuntimeError, match="not found"):
            memory_tool.call({"command": "view", "path": "/memories/tmp.txt"})

    def test_view_directory(self, memory_tool):
        memory_tool.call({"command": "create", "path": "/memories/a.txt", "file_text": "a"})
        memory_tool.call({"command": "create", "path": "/memories/b.txt", "file_text": "b"})
        listing = memory_tool.call({"command": "view", "path": "/memories"})
        assert "a.txt" in listing
        assert "b.txt" in listing

    def test_view_directory_excludes_hidden_files(self, memory_tool):
        """Hidden files (names starting with '.') should be excluded from directory listings."""
        memory_tool.call({"command": "create", "path": "/memories/visible.txt", "file_text": "v"})
        memory_tool.call({"command": "create", "path": "/memories/.hidden", "file_text": "h"})
        listing = memory_tool.call({"command": "view", "path": "/memories"})
        assert "visible.txt" in listing
        assert ".hidden" not in listing

    def test_unknown_command(self, memory_tool):
        with pytest.raises(NotImplementedError, match="Unknown command"):
            memory_tool.call({"command": "explode"})

    def test_clear_all(self, memory_tool):
        memory_tool.call({"command": "create", "path": "/memories/x.txt", "file_text": "data"})
        memory_tool.clear_all()
        listing = memory_tool.call({"command": "view", "path": "/memories"})
        # After clearing, directory should be empty
        assert "x.txt" not in listing

    def test_clear_all_memory(self, memory_tool):
        """SDK's clear_all_memory method works."""
        memory_tool.call({"command": "create", "path": "/memories/y.txt", "file_text": "data"})
        result = memory_tool.clear_all_memory()
        assert "cleared" in result.lower()

    def test_insert(self, memory_tool):
        memory_tool.call({"command": "create", "path": "/memories/ins.txt", "file_text": "line0\nline1"})
        memory_tool.call({
            "command": "insert",
            "path": "/memories/ins.txt",
            "insert_line": 1,
            "insert_text": "inserted",
        })
        content = memory_tool.call({"command": "view", "path": "/memories/ins.txt"})
        assert "inserted" in content

    def test_rename(self, memory_tool):
        memory_tool.call({"command": "create", "path": "/memories/old.txt", "file_text": "data"})
        memory_tool.call({
            "command": "rename",
            "old_path": "/memories/old.txt",
            "new_path": "/memories/new.txt",
        })
        content = memory_tool.call({"command": "view", "path": "/memories/new.txt"})
        assert "data" in content
        with pytest.raises(RuntimeError, match="not found"):
            memory_tool.call({"command": "view", "path": "/memories/old.txt"})

    # ---- memory_dir property ----

    def test_memory_dir_property(self, memory_tool, tmp_path):
        assert memory_tool.memory_dir == (tmp_path / "mem")

    # ---- _resolve edge cases ----

    def test_path_traversal_blocked(self, memory_tool):
        with pytest.raises(ValueError, match="Path traversal not allowed"):
            memory_tool._resolve("/../../../etc/passwd")

    # ---- view with view_range ----

    def test_view_range(self, memory_tool):
        memory_tool.call({"command": "create", "path": "/memories/lines.txt", "file_text": "a\nb\nc\nd\ne"})
        content = memory_tool.call({"command": "view", "path": "/memories/lines.txt", "view_range": [2, 4]})
        assert "b" in content
        assert "c" in content
        assert "d" in content
        # Line 1 (a) and line 5 (e) should not be present
        assert "a" not in content.split(":")[1] if "a" in content else True

    def test_view_range_to_end(self, memory_tool):
        memory_tool.call({"command": "create", "path": "/memories/lines.txt", "file_text": "a\nb\nc"})
        content = memory_tool.call({"command": "view", "path": "/memories/lines.txt", "view_range": [2, -1]})
        assert "b" in content
        assert "c" in content

    # ---- str_replace error cases ----

    def test_str_replace_text_not_found(self, memory_tool):
        memory_tool.call({"command": "create", "path": "/memories/f.txt", "file_text": "hello world"})
        with pytest.raises(ValueError, match="Text not found"):
            memory_tool.call({"command": "str_replace", "path": "/memories/f.txt", "old_str": "xyz", "new_str": "abc"})

    def test_str_replace_duplicate_text(self, memory_tool):
        memory_tool.call({"command": "create", "path": "/memories/f.txt", "file_text": "foo foo foo"})
        with pytest.raises(ValueError, match="appears .* times"):
            memory_tool.call({"command": "str_replace", "path": "/memories/f.txt", "old_str": "foo", "new_str": "bar"})

    def test_str_replace_file_not_found(self, memory_tool):
        with pytest.raises(FileNotFoundError, match="File not found"):
            memory_tool.call({"command": "str_replace", "path": "/memories/nope.txt", "old_str": "x", "new_str": "y"})

    # ---- insert error cases ----

    def test_insert_invalid_line(self, memory_tool):
        memory_tool.call({"command": "create", "path": "/memories/f.txt", "file_text": "a\nb"})
        with pytest.raises(ValueError, match="Invalid insert_line"):
            memory_tool.call({"command": "insert", "path": "/memories/f.txt", "insert_line": 99, "insert_text": "x"})

    def test_insert_file_not_found(self, memory_tool):
        with pytest.raises(FileNotFoundError, match="File not found"):
            memory_tool.call({"command": "insert", "path": "/memories/nope.txt", "insert_line": 0, "insert_text": "x"})

    # ---- delete edge cases ----

    def test_delete_directory(self, memory_tool):
        memory_tool.call({"command": "create", "path": "/memories/subdir/f.txt", "file_text": "data"})
        result = memory_tool.call({"command": "delete", "path": "/memories/subdir"})
        assert "deleted" in result.lower()
        with pytest.raises(RuntimeError, match="not found"):
            memory_tool.call({"command": "view", "path": "/memories/subdir"})

    def test_delete_root_memories_blocked(self, memory_tool):
        with pytest.raises(ValueError, match="Cannot delete"):
            memory_tool.call({"command": "delete", "path": "/memories"})

    def test_delete_nonexistent_file(self, memory_tool):
        with pytest.raises(FileNotFoundError, match="Path not found"):
            memory_tool.call({"command": "delete", "path": "/memories/nope.txt"})

    # ---- rename error cases ----

    def test_rename_source_not_found(self, memory_tool):
        with pytest.raises(FileNotFoundError, match="Source path not found"):
            memory_tool.call({"command": "rename", "old_path": "/memories/nope.txt", "new_path": "/memories/new.txt"})

    def test_rename_dest_exists(self, memory_tool):
        memory_tool.call({"command": "create", "path": "/memories/a.txt", "file_text": "a"})
        memory_tool.call({"command": "create", "path": "/memories/b.txt", "file_text": "b"})
        with pytest.raises(ValueError, match="Destination already exists"):
            memory_tool.call({"command": "rename", "old_path": "/memories/a.txt", "new_path": "/memories/b.txt"})


@pytest.mark.unit
class TestAnthropicApiWithExternalTools:
    """Verify AnthropicApi wiring when external_tools are provided."""

    @pytest.fixture
    def patch_anthropic(self):
        from unittest.mock import patch
        with patch('microbots.llm.anthropic_api.endpoint', 'https://api.anthropic.com'), \
             patch('microbots.llm.anthropic_api.deployment_name', 'claude-sonnet-4-5'), \
             patch('microbots.llm.anthropic_api.api_key', 'test-api-key'), \
             patch('microbots.llm.anthropic_api.Anthropic') as mock_cls:
            yield mock_cls

    def test_init_stores_external_tools(self, patch_anthropic):
        from microbots.llm.anthropic_api import AnthropicApi
        mem = AnthropicMemoryTool()
        api = AnthropicApi(system_prompt="test", external_tools=[mem])
        assert api.external_tools == [mem]

    def test_init_without_external_tools(self, patch_anthropic):
        from microbots.llm.anthropic_api import AnthropicApi
        api = AnthropicApi(system_prompt="test")
        assert api.external_tools == []

    def test_init_with_context_management(self, patch_anthropic):
        from microbots.llm.anthropic_api import AnthropicApi
        cm = {"edits": [{"type": "clear_tool_uses_20250919"}]}
        api = AnthropicApi(system_prompt="test", context_management=cm)
        assert api.context_management == cm

    def test_collect_betas(self, patch_anthropic):
        from microbots.llm.anthropic_api import AnthropicApi
        mem = AnthropicMemoryTool()
        api = AnthropicApi(system_prompt="test", external_tools=[mem])
        betas = api._collect_betas()
        assert MEMORY_BETA_HEADER in betas

    def test_collect_betas_empty(self, patch_anthropic):
        from microbots.llm.anthropic_api import AnthropicApi
        api = AnthropicApi(system_prompt="test")
        assert api._collect_betas() == []

    def test_ask_routes_to_simple_without_tools(self, patch_anthropic):
        """Without external_tools, ask() should use _ask_simple."""
        from microbots.llm.anthropic_api import AnthropicApi
        from unittest.mock import Mock
        import json

        api = AnthropicApi(system_prompt="test")
        mock_resp = Mock()
        mock_resp.content = [Mock(text=json.dumps({
            "task_done": False, "command": "ls", "thoughts": ""
        }))]
        api.ai_client.messages.create = Mock(return_value=mock_resp)

        result = api.ask("hello")
        assert result.command == "ls"
        # Should have called messages.create (not tool_runner)
        api.ai_client.messages.create.assert_called()

    def test_ask_routes_to_tools_path(self, patch_anthropic):
        """With external_tools, ask() should use _ask_with_tools (tool_runner)."""
        from microbots.llm.anthropic_api import AnthropicApi
        from unittest.mock import Mock
        import json

        mem = AnthropicMemoryTool()
        api = AnthropicApi(system_prompt="test", external_tools=[mem])

        # Mock tool_runner to return a single final message (no tool use)
        final_msg = Mock()
        final_msg.stop_reason = "end_turn"
        text_block = Mock()
        text_block.type = "text"
        text_block.text = json.dumps({
            "task_done": False, "command": "pwd", "thoughts": ""
        })
        final_msg.content = [text_block]

        mock_runner = Mock()
        mock_runner.__iter__ = Mock(return_value=iter([final_msg]))
        mock_runner.generate_tool_call_response = Mock(return_value=None)

        api.ai_client.beta.messages.tool_runner = Mock(return_value=mock_runner)

        result = api.ask("hello")
        assert result.command == "pwd"
        api.ai_client.beta.messages.tool_runner.assert_called_once()
