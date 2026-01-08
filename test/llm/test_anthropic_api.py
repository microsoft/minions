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
        assert len(api.messages) == 1  # System prompt is included in messages
        assert api.messages[0]["role"] == "system"
        assert api.messages[0]["content"] == system_prompt

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
        assert len(api.messages) == 3  # system + user + assistant
        assert api.messages[0]["role"] == "system"
        assert api.messages[1]["role"] == "user"
        assert api.messages[1]["content"] == message
        assert api.messages[2]["role"] == "assistant"

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

        # Add some messages (system message already exists)
        api.messages.append({"role": "user", "content": "Hello"})
        api.messages.append({"role": "assistant", "content": "Hi there"})
        api.messages.append({"role": "user", "content": "How are you?"})

        assert len(api.messages) == 4  # system + 3 added messages

        # Clear history
        result = api.clear_history()

        # Verify messages are empty after clear
        assert result is True
        assert len(api.messages) == 0

    def test_clear_history_returns_true(self):
        """Test that clear_history returns True"""
        system_prompt = "You are a helpful assistant"
        api = AnthropicApi(system_prompt=system_prompt)

        result = api.clear_history()

        assert result is True

    def test_clear_history_preserves_system_prompt_attribute(self):
        """Test that clear_history preserves the original system_prompt attribute"""
        system_prompt = "You are a code assistant specialized in Python"
        api = AnthropicApi(system_prompt=system_prompt)

        # Add and clear messages multiple times
        for i in range(3):
            api.messages.append({"role": "user", "content": f"Message {i}"})
            api.clear_history()

        # Verify system_prompt attribute is still correct
        assert api.system_prompt == system_prompt
        # Note: clear_history empties all messages including system
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
        assert api.messages[1]["content"] == ""  # User message (after system)

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
        # Should have: 1 system + 3 user + 3 assistant = 7 messages
        assert len(api.messages) == 7

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
        assert len(api.messages) == 0  # clear_history empties all messages


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

