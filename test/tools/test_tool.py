"""
Unit tests for the Tool class and related functions.
Tests handling of optional arguments.
"""
import os
import sys

import pytest

# Add src directory to path to import from local source
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

from microbots.tools.tool import Tool, parse_tool_definition


@pytest.mark.unit
class TestToolOptionalArguments:
    """Unit tests for Tool class optional arguments handling."""

    def test_tool_without_parameters(self):
        """Test that Tool can be created without parameters field."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            usage_instructions_to_llm="Test instructions",
            install_commands=["echo test"],
        )
        assert tool.name == "test_tool"
        assert tool.parameters is None
        assert tool.env_variables is None

    def test_tool_with_parameters(self):
        """Test that Tool can be created with parameters field."""
        params = {"type": "str", "required": True}
        tool = Tool(
            name="test_tool",
            description="A test tool",
            usage_instructions_to_llm="Test instructions",
            install_commands=["echo test"],
            parameters=params,
        )
        assert tool.parameters == params

    def test_tool_with_env_variables(self):
        """Test that Tool can be created with env_variables field."""
        env_vars = ["VAR1", "VAR2"]
        tool = Tool(
            name="test_tool",
            description="A test tool",
            usage_instructions_to_llm="Test instructions",
            install_commands=["echo test"],
            env_variables=env_vars,
        )
        assert tool.env_variables == env_vars

    def test_tool_with_verify_commands_none(self):
        """Test that Tool can be created with verify_commands set to None."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            usage_instructions_to_llm="Test instructions",
            install_commands=["echo test"],
            verify_commands=None,
        )
        assert tool.verify_commands is None

    def test_tool_with_all_optional_fields_none(self):
        """Test that Tool can be created with all optional fields as None."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            usage_instructions_to_llm="Test instructions",
            install_commands=["echo test"],
            parameters=None,
            env_variables=None,
            files_to_copy=None,
            verify_commands=None,
            setup_commands=None,
            uninstall_commands=None,
        )
        assert tool.parameters is None
        assert tool.env_variables is None
        assert tool.files_to_copy is None
        assert tool.verify_commands is None
        assert tool.setup_commands is None
        assert tool.uninstall_commands is None


@pytest.mark.unit
class TestParseToolDefinition:
    """Unit tests for parse_tool_definition function."""

    def test_parse_cscope_yaml_without_parameters(self):
        """Test parsing cscope.yaml which doesn't have parameters or env_variables."""
        tool = parse_tool_definition("cscope.yaml")
        assert tool.name == "cscope"
        assert tool.parameters is None
        assert tool.env_variables is None
        assert tool.verify_commands is not None  # cscope.yaml has verify_commands

    def test_parse_browser_use_yaml_with_parameters(self):
        """Test parsing browser-use.yaml which has parameters and env_variables."""
        tool = parse_tool_definition("browser-use.yaml")
        assert tool.name == "browser-use"
        assert tool.parameters is not None
        assert tool.env_variables is not None
        assert len(tool.env_variables) > 0


@pytest.mark.unit
class TestEnvVariablesIteration:
    """Unit tests for env_variables iteration handling."""

    def test_iterate_none_env_variables(self):
        """Test that iterating over None env_variables with 'or []' pattern works."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            usage_instructions_to_llm="Test instructions",
            install_commands=["echo test"],
            env_variables=None,
        )
        # This should not raise an exception
        count = 0
        for _ in tool.env_variables or []:
            count += 1
        assert count == 0

    def test_iterate_empty_env_variables(self):
        """Test that iterating over empty list env_variables works."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            usage_instructions_to_llm="Test instructions",
            install_commands=["echo test"],
            env_variables=[],
        )
        count = 0
        for _ in tool.env_variables or []:
            count += 1
        assert count == 0

    def test_iterate_with_env_variables(self):
        """Test that iterating over env_variables with values works."""
        env_vars = ["VAR1", "VAR2", "VAR3"]
        tool = Tool(
            name="test_tool",
            description="A test tool",
            usage_instructions_to_llm="Test instructions",
            install_commands=["echo test"],
            env_variables=env_vars,
        )
        collected = []
        for var in tool.env_variables or []:
            collected.append(var)
        assert collected == env_vars
