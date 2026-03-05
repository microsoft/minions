"""
Unit tests for MicroBot._upgrade_tools_for_provider.

These tests verify that plain ``MemoryTool`` instances are automatically
replaced with ``AnthropicMemoryTool`` when the model provider is Anthropic,
and that no changes are made for other providers or other tool types.

All tests bypass the heavy MicroBot constructor (Docker environment, LLM
creation) by constructing an uninitialized instance with ``object.__new__``
and manually setting only the attributes the method under test needs.
"""
import sys
import os
import logging
import pytest
from unittest.mock import patch, Mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from microbots.MicroBot import MicroBot
from microbots.constants import ModelProvider
from microbots.tools.tool_definitions.memory_tool import MemoryTool
from microbots.tools.tool_definitions.anthropic_memory_tool import AnthropicMemoryTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bare_microbot(model_provider: str, tools: list) -> MicroBot:
    """Return an uninitialized MicroBot with only the attributes that
    ``_upgrade_tools_for_provider`` inspects."""
    bot = object.__new__(MicroBot)
    bot.model_provider = model_provider
    bot.additional_tools = list(tools)
    return bot


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

    # -- Anthropic provider: MemoryTool → AnthropicMemoryTool ---------------

    def test_memory_tool_is_replaced_with_anthropic_variant(self, tmp_path):
        tool = _memory_tool(tmp_path)
        bot = _bare_microbot(ModelProvider.ANTHROPIC, [tool])

        bot._upgrade_tools_for_provider()

        assert len(bot.additional_tools) == 1
        assert isinstance(bot.additional_tools[0], AnthropicMemoryTool)

    def test_memory_dir_is_forwarded_to_upgraded_tool(self, tmp_path):
        mem_dir = str(tmp_path / "my_memory")
        tool = MemoryTool(memory_dir=mem_dir)
        bot = _bare_microbot(ModelProvider.ANTHROPIC, [tool])

        bot._upgrade_tools_for_provider()

        upgraded = bot.additional_tools[0]
        assert isinstance(upgraded, AnthropicMemoryTool)
        assert str(upgraded.memory_dir) == mem_dir

    def test_usage_instructions_are_forwarded_to_upgraded_tool(self, tmp_path):
        custom_instructions = "custom memory instructions for test"
        tool = _memory_tool(tmp_path, instructions=custom_instructions)
        bot = _bare_microbot(ModelProvider.ANTHROPIC, [tool])

        bot._upgrade_tools_for_provider()

        upgraded = bot.additional_tools[0]
        assert upgraded.usage_instructions_to_llm == custom_instructions

    def test_already_anthropic_memory_tool_is_not_re_upgraded(self, tmp_path):
        existing = AnthropicMemoryTool(memory_dir=str(tmp_path / "memory"))
        bot = _bare_microbot(ModelProvider.ANTHROPIC, [existing])

        bot._upgrade_tools_for_provider()

        assert len(bot.additional_tools) == 1
        assert bot.additional_tools[0] is existing

    def test_non_memory_tools_are_kept_unchanged(self, tmp_path):
        other_tool = Mock()
        other_tool.__class__ = Mock  # not a MemoryTool subclass
        bot = _bare_microbot(ModelProvider.ANTHROPIC, [other_tool])

        bot._upgrade_tools_for_provider()

        assert len(bot.additional_tools) == 1
        assert bot.additional_tools[0] is other_tool

    def test_mixed_tool_list_upgrades_only_memory_tools(self, tmp_path):
        plain_memory = _memory_tool(tmp_path)
        already_upgraded = AnthropicMemoryTool(memory_dir=str(tmp_path / "memory2"))
        other_tool = Mock(spec=[])
        bot = _bare_microbot(ModelProvider.ANTHROPIC, [plain_memory, already_upgraded, other_tool])

        bot._upgrade_tools_for_provider()

        assert len(bot.additional_tools) == 3
        # first: should have been upgraded
        assert isinstance(bot.additional_tools[0], AnthropicMemoryTool)
        assert bot.additional_tools[0] is not plain_memory
        # second: already AnthropicMemoryTool, untouched
        assert bot.additional_tools[1] is already_upgraded
        # third: non-memory tool, untouched
        assert bot.additional_tools[2] is other_tool

    def test_empty_tool_list_is_a_no_op(self):
        bot = _bare_microbot(ModelProvider.ANTHROPIC, [])

        bot._upgrade_tools_for_provider()

        assert bot.additional_tools == []

    def test_logger_info_called_for_each_upgraded_tool(self, tmp_path, caplog):
        tool1 = _memory_tool(tmp_path)
        tmp_path2 = tmp_path / "sub"
        tmp_path2.mkdir()
        tool2 = _memory_tool(tmp_path2)
        bot = _bare_microbot(ModelProvider.ANTHROPIC, [tool1, tool2])

        with caplog.at_level(logging.INFO, logger=" MicroBot "):
            bot._upgrade_tools_for_provider()

        upgrade_logs = [r for r in caplog.records if "Auto-upgrading" in r.message]
        assert len(upgrade_logs) == 2

    # -- Non-Anthropic providers: no upgrade should happen ------------------

    @pytest.mark.parametrize("provider", [ModelProvider.OPENAI, ModelProvider.OLLAMA_LOCAL])
    def test_no_upgrade_for_non_anthropic_provider(self, tmp_path, provider):
        tool = _memory_tool(tmp_path)
        bot = _bare_microbot(provider, [tool])

        bot._upgrade_tools_for_provider()

        assert len(bot.additional_tools) == 1
        assert isinstance(bot.additional_tools[0], MemoryTool)
        assert not isinstance(bot.additional_tools[0], AnthropicMemoryTool)

    @pytest.mark.parametrize("provider", [ModelProvider.OPENAI, ModelProvider.OLLAMA_LOCAL])
    def test_original_tool_identity_preserved_for_non_anthropic(self, tmp_path, provider):
        tool = _memory_tool(tmp_path)
        bot = _bare_microbot(provider, [tool])

        bot._upgrade_tools_for_provider()

        assert bot.additional_tools[0] is tool
