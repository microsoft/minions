"""
Unit tests for the tool module.
Tests TOOLTYPE enum and ToolAbstract base class.
"""
import os
import sys
from unittest.mock import Mock

import pytest

# Add src directory to path to import from local source
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

from microbots.tools.tool import TOOLTYPE, ToolAbstract
from microbots.environment.Environment import Environment


# Concrete implementation of ToolAbstract for testing
class ConcreteTool(ToolAbstract):
    """Concrete implementation of ToolAbstract for testing purposes."""

    def install_tool(self, env: Environment):
        pass

    def verify_tool_installation(self, env: Environment):
        pass

    def setup_tool(self, env: Environment):
        pass

    def uninstall_tool(self, env: Environment):
        pass


@pytest.mark.unit
class TestToolType:
    """Unit tests for TOOLTYPE enum."""

    def test_internal_value(self):
        """Test TOOLTYPE.INTERNAL has correct value."""
        assert TOOLTYPE.INTERNAL.value == "internal"

    def test_external_value(self):
        """Test TOOLTYPE.EXTERNAL has correct value."""
        assert TOOLTYPE.EXTERNAL.value == "external"

    def test_tooltype_is_string_enum(self):
        """Test TOOLTYPE inherits from str."""
        assert isinstance(TOOLTYPE.INTERNAL, str)
        assert isinstance(TOOLTYPE.EXTERNAL, str)

    def test_tooltype_string_comparison(self):
        """Test TOOLTYPE can be compared with strings."""
        assert TOOLTYPE.INTERNAL == "internal"
        assert TOOLTYPE.EXTERNAL == "external"

    def test_tooltype_members(self):
        """Test TOOLTYPE has exactly two members."""
        assert len(TOOLTYPE) == 2
        assert TOOLTYPE.INTERNAL in TOOLTYPE
        assert TOOLTYPE.EXTERNAL in TOOLTYPE


@pytest.mark.unit
class TestToolAbstractRequiredFields:
    """Unit tests for ToolAbstract required fields."""

    def test_create_tool_with_required_fields(self):
        """Test creating a tool with all required fields."""
        tool = ConcreteTool(
            name="test_tool",
            description="A test tool",
            usage_instructions_to_llm="Test instructions",
            install_commands=["echo test"],
            tool_type=TOOLTYPE.INTERNAL,
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.usage_instructions_to_llm == "Test instructions"
        assert tool.install_commands == ["echo test"]
        assert tool.tool_type == TOOLTYPE.INTERNAL

    def test_tool_with_external_type(self):
        """Test creating a tool with EXTERNAL tool_type."""
        tool = ConcreteTool(
            name="external_tool",
            description="An external tool",
            usage_instructions_to_llm="External instructions",
            install_commands=["pip install something"],
            tool_type=TOOLTYPE.EXTERNAL,
        )
        assert tool.tool_type == TOOLTYPE.EXTERNAL

    def test_tool_with_multiple_install_commands(self):
        """Test creating a tool with multiple install commands."""
        commands = ["apt-get update", "apt-get install -y tool", "tool --setup"]
        tool = ConcreteTool(
            name="test_tool",
            description="A test tool",
            usage_instructions_to_llm="Test instructions",
            install_commands=commands,
            tool_type=TOOLTYPE.INTERNAL,
        )
        assert tool.install_commands == commands
        assert len(tool.install_commands) == 3


@pytest.mark.unit
class TestToolAbstractOptionalFields:
    """Unit tests for ToolAbstract optional fields."""

    def test_default_parameters_is_empty_dict(self):
        """Test that parameters defaults to empty dict."""
        tool = ConcreteTool(
            name="test_tool",
            description="A test tool",
            usage_instructions_to_llm="Test instructions",
            install_commands=["echo test"],
            tool_type=TOOLTYPE.INTERNAL,
        )
        assert tool.parameters == {}

    def test_default_env_variables_is_empty_list(self):
        """Test that env_variables defaults to empty list."""
        tool = ConcreteTool(
            name="test_tool",
            description="A test tool",
            usage_instructions_to_llm="Test instructions",
            install_commands=["echo test"],
            tool_type=TOOLTYPE.INTERNAL,
        )
        assert tool.env_variables == []

    def test_default_verify_commands_is_empty_list(self):
        """Test that verify_commands defaults to empty list."""
        tool = ConcreteTool(
            name="test_tool",
            description="A test tool",
            usage_instructions_to_llm="Test instructions",
            install_commands=["echo test"],
            tool_type=TOOLTYPE.INTERNAL,
        )
        assert tool.verify_commands == []

    def test_default_setup_commands_is_empty_list(self):
        """Test that setup_commands defaults to empty list."""
        tool = ConcreteTool(
            name="test_tool",
            description="A test tool",
            usage_instructions_to_llm="Test instructions",
            install_commands=["echo test"],
            tool_type=TOOLTYPE.INTERNAL,
        )
        assert tool.setup_commands == []

    def test_default_uninstall_commands_is_empty_list(self):
        """Test that uninstall_commands defaults to empty list."""
        tool = ConcreteTool(
            name="test_tool",
            description="A test tool",
            usage_instructions_to_llm="Test instructions",
            install_commands=["echo test"],
            tool_type=TOOLTYPE.INTERNAL,
        )
        assert tool.uninstall_commands == []

    def test_tool_with_custom_parameters(self):
        """Test creating a tool with custom parameters."""
        params = {"type": "string", "required": True, "default": "value"}
        tool = ConcreteTool(
            name="test_tool",
            description="A test tool",
            usage_instructions_to_llm="Test instructions",
            install_commands=["echo test"],
            tool_type=TOOLTYPE.INTERNAL,
            parameters=params,
        )
        assert tool.parameters == params

    def test_tool_with_env_variables(self):
        """Test creating a tool with env_variables."""
        env_vars = ["API_KEY", "SECRET_TOKEN", "CONFIG_PATH"]
        tool = ConcreteTool(
            name="test_tool",
            description="A test tool",
            usage_instructions_to_llm="Test instructions",
            install_commands=["echo test"],
            tool_type=TOOLTYPE.INTERNAL,
            env_variables=env_vars,
        )
        assert tool.env_variables == env_vars

    def test_tool_with_all_optional_fields(self):
        """Test creating a tool with all optional fields set."""
        tool = ConcreteTool(
            name="test_tool",
            description="A test tool",
            usage_instructions_to_llm="Test instructions",
            install_commands=["echo install"],
            tool_type=TOOLTYPE.INTERNAL,
            parameters={"key": "value"},
            env_variables=["VAR1", "VAR2"],
            verify_commands=["echo verify"],
            setup_commands=["echo setup"],
            uninstall_commands=["echo uninstall"],
        )
        assert tool.parameters == {"key": "value"}
        assert tool.env_variables == ["VAR1", "VAR2"]
        assert tool.verify_commands == ["echo verify"]
        assert tool.setup_commands == ["echo setup"]
        assert tool.uninstall_commands == ["echo uninstall"]


@pytest.mark.unit
class TestToolAbstractMethods:
    """Unit tests for ToolAbstract methods."""

    def test_is_model_supported_default_returns_true(self):
        """Test that is_model_supported returns True by default."""
        tool = ConcreteTool(
            name="test_tool",
            description="A test tool",
            usage_instructions_to_llm="Test instructions",
            install_commands=["echo test"],
            tool_type=TOOLTYPE.INTERNAL,
        )
        assert tool.is_model_supported("gpt-4") is True
        assert tool.is_model_supported("claude-3") is True
        assert tool.is_model_supported("any-model-name") is True

    def test_is_model_supported_with_empty_string(self):
        """Test is_model_supported with empty string."""
        tool = ConcreteTool(
            name="test_tool",
            description="A test tool",
            usage_instructions_to_llm="Test instructions",
            install_commands=["echo test"],
            tool_type=TOOLTYPE.INTERNAL,
        )
        assert tool.is_model_supported("") is True

    def test_abstract_methods_can_be_called(self):
        """Test that abstract methods can be called on concrete implementation."""
        tool = ConcreteTool(
            name="test_tool",
            description="A test tool",
            usage_instructions_to_llm="Test instructions",
            install_commands=["echo test"],
            tool_type=TOOLTYPE.INTERNAL,
        )
        mock_env = Mock()

        # These should not raise exceptions
        tool.install_tool(mock_env)
        tool.verify_tool_installation(mock_env)
        tool.setup_tool(mock_env)
        tool.uninstall_tool(mock_env)


@pytest.mark.unit
class TestToolAbstractIsAbstract:
    """Unit tests verifying ToolAbstract is properly abstract."""

    def test_cannot_instantiate_tool_abstract_directly(self):
        """Test that ToolAbstract cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ToolAbstract(
                name="test_tool",
                description="A test tool",
                usage_instructions_to_llm="Test instructions",
                install_commands=["echo test"],
                tool_type=TOOLTYPE.INTERNAL,
            )


@pytest.mark.unit
class TestToolAbstractInheritance:
    """Unit tests for ToolAbstract inheritance behavior."""

    def test_concrete_tool_is_instance_of_tool_abstract(self):
        """Test that concrete implementation is instance of ToolAbstract."""
        tool = ConcreteTool(
            name="test_tool",
            description="A test tool",
            usage_instructions_to_llm="Test instructions",
            install_commands=["echo test"],
            tool_type=TOOLTYPE.INTERNAL,
        )
        assert isinstance(tool, ToolAbstract)

    def test_subclass_can_override_is_model_supported(self):
        """Test that subclass can override is_model_supported method."""

        class SelectiveModelTool(ToolAbstract):
            def is_model_supported(self, model_name: str) -> bool:
                return model_name.startswith("gpt")

            def install_tool(self, env):
                pass

            def verify_tool_installation(self, env):
                pass

            def setup_tool(self, env):
                pass

            def uninstall_tool(self, env):
                pass

        tool = SelectiveModelTool(
            name="selective_tool",
            description="A tool that only supports GPT models",
            usage_instructions_to_llm="Test instructions",
            install_commands=["echo test"],
            tool_type=TOOLTYPE.INTERNAL,
        )
        assert tool.is_model_supported("gpt-4") is True
        assert tool.is_model_supported("gpt-3.5-turbo") is True
        assert tool.is_model_supported("claude-3") is False
