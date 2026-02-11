"""
Unit tests for the Tool hierarchy and ExternalTool base class.
"""
import pytest

from anthropic.lib.tools import BetaAbstractMemoryTool as _SDKMemoryTool

from microbots.tools.tool import Tool, InternalTool
from microbots.tools.external_tool import ExternalTool
from microbots.tools.external.anthropic_memory_tool import AnthropicMemoryTool


@pytest.mark.unit
class TestToolHierarchy:
    """Verify the ABC / inheritance relationships."""

    def test_tool_is_base_class(self):
        """Tool serves as the base class for all tools."""
        assert issubclass(InternalTool, Tool)
        assert issubclass(ExternalTool, Tool)

    def test_external_tool_is_abstract(self):
        """ExternalTool ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ExternalTool()  # type: ignore[abstract]

    def test_internal_tool_is_subclass_of_tool(self):
        assert issubclass(InternalTool, Tool)

    def test_external_tool_is_subclass_of_tool(self):
        assert issubclass(ExternalTool, Tool)

    def test_memory_tool_is_subclass_of_external_tool(self):
        assert issubclass(AnthropicMemoryTool, ExternalTool)

    def test_memory_tool_is_subclass_of_tool(self):
        assert issubclass(AnthropicMemoryTool, Tool)

    def test_memory_tool_is_subclass_of_sdk_memory(self):
        assert issubclass(AnthropicMemoryTool, _SDKMemoryTool)

    def test_internal_tool_instance_is_tool(self):
        t = InternalTool(
            name="x", description="y",
            usage_instructions_to_llm="z", install_commands=["echo 1"],
        )
        assert isinstance(t, Tool)

    def test_memory_tool_instance_is_tool(self):
        m = AnthropicMemoryTool()
        assert isinstance(m, Tool)

    def test_memory_tool_instance_is_external_tool(self):
        m = AnthropicMemoryTool()
        assert isinstance(m, ExternalTool)

