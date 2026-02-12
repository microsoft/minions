"""
Unit tests for AnthropicApi class
"""
import pytest
import json
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from microbots.llm.anthropic_api import AnthropicApi
from microbots.llm.llm import LLMAskResponse, LLMInterface, llm_output_format_str


@pytest.fixture
def patch_anthropic_config(request):
    """Patch Anthropic configuration for unit tests only"""
    # Skip patching for integration tests
    if 'anthropic_integration' in request.keywords:
        yield None
    else:
        with patch('microbots.llm.anthropic_api.endpoint', 'https://api.anthropic.com'), \
             patch('microbots.llm.anthropic_api.deployment_name', 'claude-sonnet-4-5'), \
             patch('microbots.llm.anthropic_api.api_key', 'test-api-key'), \
             patch('microbots.llm.anthropic_api.Anthropic') as mock_anthropic:
            yield mock_anthropic


@pytest.mark.unit
class TestAnthropicApiInitialization:
    """Tests for AnthropicApi initialization"""

    @pytest.fixture(autouse=True)
    def _use_patch(self, patch_anthropic_config):
        """Apply patch for unit tests"""
        pass

    def test_init_with_default_deployment_name(self):
        """Test initialization with deployment name from parameter default"""
        system_prompt = "You are a helpful assistant"

        api = AnthropicApi(system_prompt=system_prompt)

        assert api.system_prompt == system_prompt
        assert api.max_retries == 3
        assert api.retries == 0
        assert len(api.messages) == 0  # Anthropic doesn't include system in messages

    def test_init_with_custom_deployment_name(self):
        """Test initialization with custom deployment name"""
        system_prompt = "You are a helpful assistant"
        custom_deployment = "claude-3-opus"

        api = AnthropicApi(
            system_prompt=system_prompt,
            deployment_name=custom_deployment
        )

        assert api.deployment_name == custom_deployment

    def test_init_with_custom_max_retries(self):
        """Test initialization with custom max_retries"""
        system_prompt = "You are a helpful assistant"

        api = AnthropicApi(
            system_prompt=system_prompt,
            max_retries=5
        )

        assert api.max_retries == 5
        assert api.retries == 0

    def test_init_creates_anthropic_client(self):
        """Test that initialization creates Anthropic client"""
        system_prompt = "You are a helpful assistant"

        api = AnthropicApi(system_prompt=system_prompt)

        assert api.ai_client is not None


@pytest.mark.unit
class TestAnthropicApiAsk:
    """Tests for AnthropicApi.ask method"""

    @pytest.fixture(autouse=True)
    def _use_patch(self, patch_anthropic_config):
        """Apply patch for unit tests"""
        pass

    def test_ask_successful_response(self):
        """Test ask method with successful response"""
        system_prompt = "You are a helpful assistant"
        api = AnthropicApi(system_prompt=system_prompt)

        # Mock the Anthropic client response
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = json.dumps({
            "task_done": False,
            "command": "echo 'hello'",
            "thoughts": ""
        })
        mock_response.content = [mock_content]
        api.ai_client.messages.create = Mock(return_value=mock_response)

        # Call ask
        message = "Please say hello"
        result = api.ask(message)

        # Verify the result
        assert isinstance(result, LLMAskResponse)
        assert result.task_done is False
        assert result.command == "echo 'hello'"
        assert result.thoughts == ""

        # Verify retries was reset
        assert api.retries == 0

        # Verify messages were appended
        assert len(api.messages) == 2  # user + assistant (no system in messages)
        assert api.messages[0]["role"] == "user"
        assert api.messages[0]["content"] == message
        assert api.messages[1]["role"] == "assistant"

    def test_ask_with_task_done_true(self):
        """Test ask method when task is complete"""
        system_prompt = "You are a helpful assistant"
        api = AnthropicApi(system_prompt=system_prompt)

        # Mock the Anthropic client response
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = json.dumps({
            "task_done": True,
            "command": "",
            "thoughts": "Task completed successfully"
        })
        mock_response.content = [mock_content]
        api.ai_client.messages.create = Mock(return_value=mock_response)

        # Call ask
        result = api.ask("Complete the task")

        # Verify the result
        assert result.task_done is True
        assert result.command == ""
        assert result.thoughts == "Task completed successfully"

    def test_ask_with_retry_on_invalid_response(self):
        """Test ask method retries on invalid response then succeeds"""
        system_prompt = "You are a helpful assistant"
        api = AnthropicApi(system_prompt=system_prompt)

        # Mock the Anthropic client to return invalid then valid response
        mock_invalid_response = Mock()
        mock_invalid_content = Mock()
        mock_invalid_content.text = "invalid json"
        mock_invalid_response.content = [mock_invalid_content]

        mock_valid_response = Mock()
        mock_valid_content = Mock()
        mock_valid_content.text = json.dumps({
            "task_done": False,
            "command": "ls -la",
            "thoughts": ""
        })
        mock_valid_response.content = [mock_valid_content]

        api.ai_client.messages.create = Mock(
            side_effect=[mock_invalid_response, mock_valid_response]
        )

        # Call ask
        result = api.ask("List files")

        # Verify it eventually succeeded
        assert result.task_done is False
        assert result.command == "ls -la"

        # Verify it called the API twice (retry happened)
        assert api.ai_client.messages.create.call_count == 2

    def test_ask_appends_user_message(self):
        """Test that ask appends user message to messages list"""
        system_prompt = "You are a helpful assistant"
        api = AnthropicApi(system_prompt=system_prompt)

        initial_message_count = len(api.messages)

        # Mock the Anthropic client response
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = json.dumps({
            "task_done": False,
            "command": "pwd",
            "thoughts": ""
        })
        mock_response.content = [mock_content]
        api.ai_client.messages.create = Mock(return_value=mock_response)

        # Call ask
        user_message = "What directory am I in?"
        api.ask(user_message)

        # Verify user message was added
        assert len(api.messages) > initial_message_count
        user_messages = [m for m in api.messages if m["role"] == "user"]
        assert user_messages[-1]["content"] == user_message

    def test_ask_appends_assistant_response_as_json(self):
        """Test that ask appends assistant response as JSON string"""
        system_prompt = "You are a helpful assistant"
        api = AnthropicApi(system_prompt=system_prompt)

        # Mock the Anthropic client response
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = json.dumps({
            "task_done": False,
            "command": "echo test",
            "thoughts": ""
        })
        mock_response.content = [mock_content]
        api.ai_client.messages.create = Mock(return_value=mock_response)

        # Call ask
        api.ask("Run echo test")

        # Verify assistant message was added as JSON
        assistant_messages = [m for m in api.messages if m["role"] == "assistant"]
        assert len(assistant_messages) > 0

        # Parse the assistant message to verify it's valid JSON
        assistant_content = json.loads(assistant_messages[-1]["content"])
        assert assistant_content["task_done"] is False
        assert assistant_content["command"] == "echo test"
        assert assistant_content["thoughts"] == ""

    def test_ask_uses_asdict_for_response(self):
        """Test that ask uses asdict to convert LLMAskResponse to dict"""
        system_prompt = "You are a helpful assistant"
        api = AnthropicApi(system_prompt=system_prompt)

        # Mock the Anthropic client response
        mock_response = Mock()
        mock_content = Mock()
        response_dict = {
            "task_done": True,
            "command": "",
            "thoughts": "Done"
        }
        mock_content.text = json.dumps(response_dict)
        mock_response.content = [mock_content]
        api.ai_client.messages.create = Mock(return_value=mock_response)

        # Call ask
        result = api.ask("Complete task")

        # Verify the assistant message contains the correct structure
        assistant_msg = json.loads(api.messages[-1]["content"])

        # Verify it matches what asdict would produce
        expected = asdict(result)
        assert assistant_msg == expected

    def test_ask_resets_retries_to_zero(self):
        """Test that ask resets retries to 0 at the start"""
        system_prompt = "You are a helpful assistant"
        api = AnthropicApi(system_prompt=system_prompt)

        # Set retries to a non-zero value
        api.retries = 5

        # Mock the Anthropic client response
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = json.dumps({
            "task_done": False,
            "command": "ls",
            "thoughts": ""
        })
        mock_response.content = [mock_content]
        api.ai_client.messages.create = Mock(return_value=mock_response)

        # Call ask
        api.ask("List files")

        # Verify retries was reset to 0
        assert api.retries == 0

    def test_ask_extracts_json_from_markdown(self):
        """Test that ask extracts JSON from markdown code blocks"""
        system_prompt = "You are a helpful assistant"
        api = AnthropicApi(system_prompt=system_prompt)

        # Mock response with markdown-wrapped JSON
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = """Here's the response:
```json
{
    "task_done": false,
    "command": "cat file.txt",
    "thoughts": ""
}
```"""
        mock_response.content = [mock_content]
        api.ai_client.messages.create = Mock(return_value=mock_response)

        # Call ask
        result = api.ask("Read the file")

        # Verify JSON was extracted successfully
        assert result.task_done is False
        assert result.command == "cat file.txt"


@pytest.mark.unit
class TestAnthropicApiClearHistory:
    """Tests for AnthropicApi.clear_history method"""

    @pytest.fixture(autouse=True)
    def _use_patch(self, patch_anthropic_config):
        """Apply patch for unit tests"""
        pass

    def test_clear_history_empties_messages(self):
        """Test that clear_history removes all messages"""
        system_prompt = "You are a helpful assistant"
        api = AnthropicApi(system_prompt=system_prompt)

        # Add some messages
        api.messages.append({"role": "user", "content": "Hello"})
        api.messages.append({"role": "assistant", "content": "Hi there"})
        api.messages.append({"role": "user", "content": "How are you?"})

        assert len(api.messages) == 3

        # Clear history
        result = api.clear_history()

        # Verify messages are empty (Anthropic doesn't store system in messages)
        assert result is True
        assert len(api.messages) == 0

    def test_clear_history_returns_true(self):
        """Test that clear_history returns True"""
        system_prompt = "You are a helpful assistant"
        api = AnthropicApi(system_prompt=system_prompt)

        result = api.clear_history()

        assert result is True

    def test_clear_history_preserves_system_prompt(self):
        """Test that clear_history preserves the original system prompt"""
        system_prompt = "You are a code assistant specialized in Python"
        api = AnthropicApi(system_prompt=system_prompt)

        # Add and clear messages multiple times
        for i in range(3):
            api.messages.append({"role": "user", "content": f"Message {i}"})
            api.clear_history()

        # Verify system prompt is still correct
        assert api.system_prompt == system_prompt
        assert len(api.messages) == 0


@pytest.mark.unit
class TestAnthropicApiInheritance:
    """Tests to verify AnthropicApi correctly inherits from LLMInterface"""

    @pytest.fixture(autouse=True)
    def _use_patch(self, patch_anthropic_config):
        """Apply patch for unit tests"""
        pass

    def test_anthropic_api_is_llm_interface(self):
        """Test that AnthropicApi is an instance of LLMInterface"""
        system_prompt = "You are a helpful assistant"
        api = AnthropicApi(system_prompt=system_prompt)

        assert isinstance(api, LLMInterface)

    def test_anthropic_api_implements_ask(self):
        """Test that AnthropicApi implements ask method"""
        system_prompt = "You are a helpful assistant"
        api = AnthropicApi(system_prompt=system_prompt)

        assert hasattr(api, 'ask')
        assert callable(api.ask)

    def test_anthropic_api_implements_clear_history(self):
        """Test that AnthropicApi implements clear_history method"""
        system_prompt = "You are a helpful assistant"
        api = AnthropicApi(system_prompt=system_prompt)

        assert hasattr(api, 'clear_history')
        assert callable(api.clear_history)


@pytest.mark.unit
class TestAnthropicApiEdgeCases:
    """Tests for edge cases and error scenarios"""

    @pytest.fixture(autouse=True)
    def _use_patch(self, patch_anthropic_config):
        """Apply patch for unit tests"""
        pass

    def test_ask_with_empty_message(self):
        """Test ask with empty string message"""
        system_prompt = "You are a helpful assistant"
        api = AnthropicApi(system_prompt=system_prompt)

        # Mock the Anthropic client response
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = json.dumps({
            "task_done": False,
            "command": "echo ''",
            "thoughts": ""
        })
        mock_response.content = [mock_content]
        api.ai_client.messages.create = Mock(return_value=mock_response)

        # Call ask with empty message
        result = api.ask("")

        # Verify it still works
        assert isinstance(result, LLMAskResponse)
        assert api.messages[0]["content"] == ""  # User message

    def test_multiple_ask_calls_append_messages(self):
        """Test that multiple ask calls append all messages"""
        system_prompt = "You are a helpful assistant"
        api = AnthropicApi(system_prompt=system_prompt)

        # Mock the Anthropic client response
        mock_response = Mock()
        mock_content = Mock()
        mock_content.text = json.dumps({
            "task_done": False,
            "command": "pwd",
            "thoughts": ""
        })
        mock_response.content = [mock_content]
        api.ai_client.messages.create = Mock(return_value=mock_response)

        # Make multiple ask calls
        api.ask("First question")
        api.ask("Second question")
        api.ask("Third question")

        # Verify all messages are preserved
        # Should have: 3 user + 3 assistant = 6 messages (no system in messages)
        assert len(api.messages) == 6

        user_messages = [m for m in api.messages if m["role"] == "user"]
        assert len(user_messages) == 3
        assert user_messages[0]["content"] == "First question"
        assert user_messages[1]["content"] == "Second question"
        assert user_messages[2]["content"] == "Third question"


@pytest.mark.anthropic_integration
class TestAnthropicApiIntegration:
    """Integration tests that require actual Anthropic API"""

    def test_anthropic_api_with_real_service(self):
        """Test AnthropicApi with actual Anthropic service"""
        system_prompt = "This is a capability test for you to check whether you can follow instructions properly."

        # Use real Anthropic API (requires ANTHROPIC_API_KEY in environment)
        try:
            api = AnthropicApi(system_prompt=system_prompt)
        except Exception as e:
            pytest.skip(f"Failed to initialize Anthropic API: {e}")

        # Test basic ask
        try:
            response = api.ask(f"Echo 'test' - provide a sample response in following JSON format {llm_output_format_str}")
        except Exception as e:
            pytest.skip(f"ask method raised an exception: {e}")

        assert isinstance(response, LLMAskResponse)
        assert hasattr(response, 'task_done')
        assert hasattr(response, 'command')
        assert hasattr(response, 'thoughts')

    def test_anthropic_api_clear_history_integration(self):
        """Test clear_history with actual Anthropic service"""
        system_prompt = "You are a helpful assistant"

        try:
            api = AnthropicApi(system_prompt=system_prompt)
        except Exception as e:
            pytest.skip(f"Failed to initialize Anthropic API: {e}")

        # Add some interaction
        api.messages.append({"role": "user", "content": "test"})
        api.messages.append({"role": "assistant", "content": "response"})

        # Clear history
        result = api.clear_history()

        assert result is True
        assert len(api.messages) == 0  # Anthropic doesn't store system in messages


@pytest.mark.unit
class TestAnthropicApiToolRunnerKwargs:
    """Tests that _ask_with_tools passes the correct kwargs to tool_runner."""

    @pytest.fixture(autouse=True)
    def _use_patch(self, patch_anthropic_config):
        pass

    @staticmethod
    def _make_final_message(response_json: dict):
        """Helper: build a mock final tool-runner message containing *response_json*."""
        text_block = Mock()
        text_block.type = "text"
        text_block.text = json.dumps(response_json)
        msg = Mock()
        msg.content = [text_block]
        return msg

    def _build_api_with_runner(self, external_tools, context_management=None):
        """Helper: create AnthropicApi with mocked tool_runner returning a single text message."""
        api = AnthropicApi(
            system_prompt="test",
            external_tools=external_tools,
            context_management=context_management,
        )
        final_msg = self._make_final_message(
            {"task_done": False, "command": "pwd", "thoughts": ""}
        )
        mock_runner = Mock()
        mock_runner.__iter__ = Mock(return_value=iter([final_msg]))
        mock_runner.generate_tool_call_response = Mock(return_value=None)
        api.ai_client.beta.messages.tool_runner = Mock(return_value=mock_runner)
        return api

    def test_betas_passed_to_tool_runner(self):
        """Verify that beta headers collected from tools are forwarded to tool_runner."""
        tool = Mock(spec=["beta_header"])
        tool.beta_header = "custom-beta-2025"

        api = self._build_api_with_runner(external_tools=[tool])
        api.ask("go")

        call_kwargs = api.ai_client.beta.messages.tool_runner.call_args
        assert "betas" in call_kwargs.kwargs or "betas" in (call_kwargs[1] if len(call_kwargs) > 1 else {})
        # Extract from either positional dict or keyword
        passed_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert "custom-beta-2025" in passed_kwargs["betas"]

    def test_context_management_passed_to_tool_runner(self):
        """Verify that context_management config is forwarded to tool_runner."""
        cm = {"edits": [{"type": "clear_tool_uses_20250919"}]}
        tool = Mock(spec=[])  # no beta_header

        api = self._build_api_with_runner(external_tools=[tool], context_management=cm)
        api.ask("go")

        passed_kwargs = api.ai_client.beta.messages.tool_runner.call_args.kwargs
        assert "context_management" in passed_kwargs
        assert passed_kwargs["context_management"] == cm

    def test_context_management_omitted_when_none(self):
        """context_management should NOT appear in kwargs when it is None."""
        tool = Mock(spec=[])

        api = self._build_api_with_runner(external_tools=[tool], context_management=None)
        api.ask("go")

        passed_kwargs = api.ai_client.beta.messages.tool_runner.call_args.kwargs
        assert "context_management" not in passed_kwargs


@pytest.mark.unit
class TestAnthropicApiAskWithToolsPaths:
    """Tests for code paths inside _ask_with_tools."""

    @pytest.fixture(autouse=True)
    def _use_patch(self, patch_anthropic_config):
        pass

    @staticmethod
    def _make_final_message(text):
        block = Mock()
        block.type = "text"
        block.text = text
        msg = Mock()
        msg.content = [block]
        return msg

    def test_json_extracted_from_markdown_in_tools_path(self):
        """_ask_with_tools should extract JSON wrapped in markdown code blocks."""
        tool = Mock(spec=[])
        api = AnthropicApi(system_prompt="test", external_tools=[tool])

        markdown_response = '```json\n{"task_done": false, "command": "cat f.txt", "thoughts": ""}\n```'
        final_msg = self._make_final_message(markdown_response)

        mock_runner = Mock()
        mock_runner.__iter__ = Mock(return_value=iter([final_msg]))
        mock_runner.generate_tool_call_response = Mock(return_value=None)
        api.ai_client.beta.messages.tool_runner = Mock(return_value=mock_runner)

        result = api.ask("read file")
        assert result.command == "cat f.txt"

    def test_fallback_append_when_no_prior_assistant_message(self):
        """When _build_message_content always returns [], the for/else branch should append."""
        tool = Mock(spec=[])
        api = AnthropicApi(system_prompt="test", external_tools=[tool])

        # Valid response message — _extract_text will find text, but we patch
        # _build_message_content to return [] so no assistant msg is appended
        # during the runner loop.
        valid_msg = self._make_final_message(
            json.dumps({"task_done": True, "command": "", "thoughts": "done"})
        )
        mock_runner = Mock()
        mock_runner.__iter__ = Mock(return_value=iter([valid_msg]))
        mock_runner.generate_tool_call_response = Mock(return_value=None)
        api.ai_client.beta.messages.tool_runner = Mock(return_value=mock_runner)

        with patch.object(AnthropicApi, '_build_message_content', return_value=[]):
            result = api.ask("do it")

        assert result.task_done is True
        # The for/else branch should have fired — no prior assistant message existed
        assistant_msgs = [m for m in api.messages if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 1
        last = json.loads(assistant_msgs[-1]["content"])
        assert last["task_done"] is True


@pytest.mark.unit
class TestBuildMessageContent:
    """Tests for the static _build_message_content method."""

    def test_text_block(self):
        block = Mock(type="text", text="hello")
        msg = Mock(content=[block])
        result = AnthropicApi._build_message_content(msg)
        assert result == [{"type": "text", "text": "hello"}]

    def test_tool_use_block(self):
        block = Mock(type="tool_use", id="tu_1", input={"a": 1})
        block.name = "memory"
        msg = Mock(content=[block])
        result = AnthropicApi._build_message_content(msg)
        assert result == [{"type": "tool_use", "id": "tu_1", "name": "memory", "input": {"a": 1}}]

    def test_server_tool_use_block_with_model_dump(self):
        """Blocks with unknown type that have model_dump() should be serialised via model_dump."""
        block = Mock(type="server_tool_use")
        block.model_dump = Mock(return_value={"type": "server_tool_use", "id": "stu_1", "name": "web"})
        msg = Mock(content=[block])
        result = AnthropicApi._build_message_content(msg)
        assert result == [{"type": "server_tool_use", "id": "stu_1", "name": "web"}]
        block.model_dump.assert_called_once()

    def test_unknown_block_without_model_dump(self):
        """Blocks with unknown type and no model_dump should be appended as-is."""
        block = {"type": "custom", "data": 42}
        msg = Mock(content=[block])
        result = AnthropicApi._build_message_content(msg)
        assert result == [{"type": "custom", "data": 42}]

    def test_empty_content(self):
        msg = Mock(content=[])
        result = AnthropicApi._build_message_content(msg)
        assert result == []


@pytest.mark.unit
class TestExtractText:
    """Tests for the static _extract_text method."""

    def test_returns_empty_string_for_none(self):
        assert AnthropicApi._extract_text(None) == ""

    def test_extracts_single_text_block(self):
        block = Mock(type="text", text="hello")
        resp = Mock(content=[block])
        assert AnthropicApi._extract_text(resp) == "hello"

    def test_extracts_multiple_text_blocks(self):
        b1 = Mock(type="text", text="hello")
        b2 = Mock(type="text", text="world")
        resp = Mock(content=[b1, b2])
        assert AnthropicApi._extract_text(resp) == "hello\nworld"

    def test_skips_non_text_blocks(self):
        text_block = Mock(type="text", text="result")
        tool_block = Mock(type="tool_use")
        resp = Mock(content=[tool_block, text_block])
        assert AnthropicApi._extract_text(resp) == "result"

    def test_returns_empty_string_when_no_text_blocks(self):
        tool_block = Mock(type="tool_use")
        resp = Mock(content=[tool_block])
        assert AnthropicApi._extract_text(resp) == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

