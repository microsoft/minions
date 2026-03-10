"""Unit tests for AnthropicApi.upgrade_tools() method.

These tests verify that plain ``MemoryTool`` instances are automatically
replaced with ``AnthropicMemoryTool`` when using ``AnthropicApi.upgrade_tools``.
"""
import sys
import os
import logging
import pytest
from unittest.mock import patch, Mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from microbots.llm.anthropic_api import AnthropicApi
from microbots.tools.tool_definitions.memory_tool import MemoryTool
from microbots.tools.tool_definitions.anthropic_memory_tool import AnthropicMemoryTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _memory_tool(tmp_path, instructions: str = "default instructions") -> MemoryTool:
    return MemoryTool(
        memory_dir=str(tmp_path / "memory"),
        usage_instructions_to_llm=instructions,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUpgradeToolsForProvider:

    @pytest.fixture(autouse=True)
    def _create_api(self):
        with patch("microbots.llm.anthropic_api.Anthropic"):
            self.api = AnthropicApi(system_prompt="test")

    # -- AnthropicApi.upgrade_tools: MemoryTool → AnthropicMemoryTool --------

    def test_memory_tool_is_replaced_with_anthropic_variant(self, tmp_path):
        tool = _memory_tool(tmp_path)

        upgraded = self.api.upgrade_tools([tool])

        assert len(upgraded) == 1
        assert isinstance(upgraded[0], AnthropicMemoryTool)

    def test_memory_dir_is_forwarded_to_upgraded_tool(self, tmp_path):
        mem_dir = str(tmp_path / "my_memory")
        tool = MemoryTool(memory_dir=mem_dir)

        upgraded = self.api.upgrade_tools([tool])

        assert isinstance(upgraded[0], AnthropicMemoryTool)
        assert str(upgraded[0].memory_dir) == mem_dir

    def test_usage_instructions_are_forwarded_to_upgraded_tool(self, tmp_path):
        custom_instructions = "custom memory instructions for test"
        tool = _memory_tool(tmp_path, instructions=custom_instructions)

        upgraded = self.api.upgrade_tools([tool])

        assert upgraded[0].usage_instructions_to_llm == custom_instructions

    def test_already_anthropic_memory_tool_is_not_re_upgraded(self, tmp_path):
        existing = AnthropicMemoryTool(memory_dir=str(tmp_path / "memory"))

        upgraded = self.api.upgrade_tools([existing])

        assert len(upgraded) == 1
        assert upgraded[0] is existing

    def test_non_memory_tools_are_kept_unchanged(self, tmp_path):
        other_tool = Mock()
        other_tool.__class__ = Mock  # not a MemoryTool subclass

        upgraded = self.api.upgrade_tools([other_tool])

        assert len(upgraded) == 1
        assert upgraded[0] is other_tool

    def test_mixed_tool_list_upgrades_only_memory_tools(self, tmp_path):
        plain_memory = _memory_tool(tmp_path)
        already_upgraded = AnthropicMemoryTool(memory_dir=str(tmp_path / "memory2"))
        other_tool = Mock(spec=[])

        upgraded = self.api.upgrade_tools([plain_memory, already_upgraded, other_tool])

        assert len(upgraded) == 3
        # first: should have been upgraded
        assert isinstance(upgraded[0], AnthropicMemoryTool)
        assert upgraded[0] is not plain_memory
        # second: already AnthropicMemoryTool, untouched
        assert upgraded[1] is already_upgraded
        # third: non-memory tool, untouched
        assert upgraded[2] is other_tool

    def test_empty_tool_list_is_a_no_op(self):
        upgraded = self.api.upgrade_tools([])

        assert upgraded == []

    def test_logger_info_called_for_each_upgraded_tool(self, tmp_path, caplog):
        tool1 = _memory_tool(tmp_path)
        tmp_path2 = tmp_path / "sub"
        tmp_path2.mkdir()
        tool2 = _memory_tool(tmp_path2)

        with caplog.at_level(logging.INFO):
            self.api.upgrade_tools([tool1, tool2])

        upgrade_logs = [r for r in caplog.records if "Auto-upgrading" in r.message]
        assert len(upgrade_logs) == 2
