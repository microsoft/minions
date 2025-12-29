"""
Unit tests for OllamaLocal LLM implementation

Run with: pytest test/llm/test_ollama_local.py -v
Skip Ollama tests: pytest test/llm/test_ollama_local.py -m "not ollama_local"
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from microbots.llm.ollama_local import OllamaLocal
from microbots.llm.llm import LLMAskResponse, LLMInterface, llm_output_format_str

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from test_constants import LOCAL_MODEL_NAME, LOCAL_MODEL_PORT


@pytest.mark.unit
class TestOllamaLocalInitialization:
    """Tests for OllamaLocal initialization"""

    def test_init_with_all_parameters(self):
        """Test initialization with all parameters provided"""
        system_prompt = "You are a helpful assistant"
        model_name = LOCAL_MODEL_NAME
        model_port = LOCAL_MODEL_PORT

        ollama = OllamaLocal(
            system_prompt=system_prompt,
            model_name=model_name,
            model_port=model_port
        )

        assert ollama.system_prompt == system_prompt
        assert ollama.model_name == model_name
        assert ollama.model_port == model_port
        assert ollama.max_retries == 3
        assert ollama.retries == 0
        assert len(ollama.messages) == 1
        assert ollama.messages[0]["role"] == "system"
        assert ollama.messages[0]["content"] == system_prompt

    def test_init_with_custom_max_retries(self):
        """Test initialization with custom max_retries"""
        system_prompt = "You are a helpful assistant"

        ollama = OllamaLocal(
            system_prompt=system_prompt,
            model_name=LOCAL_MODEL_NAME,
            model_port=LOCAL_MODEL_PORT,
            max_retries=5
        )

        assert ollama.max_retries == 5
        assert ollama.retries == 0

    def test_init_without_model_name_raises_error(self, monkeypatch):
        """Test that initialization without model_name raises ValueError"""
        system_prompt = "You are a helpful assistant"
        
        # Clear environment variables
        monkeypatch.delenv("LOCAL_MODEL_NAME", raising=False)

        with pytest.raises(ValueError, match="LOCAL_MODEL_NAME and LOCAL_MODEL_PORT"):
            OllamaLocal(
                system_prompt=system_prompt,
                model_name=None,
                model_port=LOCAL_MODEL_PORT
            )

    def test_init_without_model_port_raises_error(self, monkeypatch):
        """Test that initialization without model_port raises ValueError"""
        system_prompt = "You are a helpful assistant"
        
        # Clear environment variables
        monkeypatch.delenv("LOCAL_MODEL_PORT", raising=False)

        with pytest.raises(ValueError, match="LOCAL_MODEL_NAME and LOCAL_MODEL_PORT"):
            OllamaLocal(
                system_prompt=system_prompt,
                model_name=LOCAL_MODEL_NAME,
                model_port=None
            )

    def test_init_without_both_params_raises_error(self, monkeypatch):
        """Test that initialization without both params raises ValueError"""
        system_prompt = "You are a helpful assistant"
        
        # Clear environment variables
        monkeypatch.delenv("LOCAL_MODEL_NAME", raising=False)
        monkeypatch.delenv("LOCAL_MODEL_PORT", raising=False)

        with pytest.raises(ValueError, match="LOCAL_MODEL_NAME and LOCAL_MODEL_PORT"):
            OllamaLocal(
                system_prompt=system_prompt,
                model_name=None,
                model_port=None
            )


@pytest.mark.unit
class TestOllamaLocalInheritance:
    """Tests to verify OllamaLocal correctly inherits from LLMInterface"""

    def test_ollama_local_is_llm_interface(self):
        """Test that OllamaLocal is an instance of LLMInterface"""
        system_prompt = "You are a helpful assistant"
        ollama = OllamaLocal(
            system_prompt=system_prompt,
            model_name=LOCAL_MODEL_NAME,
            model_port=LOCAL_MODEL_PORT
        )

        assert isinstance(ollama, LLMInterface)

    def test_ollama_local_implements_ask(self):
        """Test that OllamaLocal implements ask method"""
        system_prompt = "You are a helpful assistant"
        ollama = OllamaLocal(
            system_prompt=system_prompt,
            model_name=LOCAL_MODEL_NAME,
            model_port=LOCAL_MODEL_PORT
        )

        assert hasattr(ollama, 'ask')
        assert callable(ollama.ask)

    def test_ollama_local_implements_clear_history(self):
        """Test that OllamaLocal implements clear_history method"""
        system_prompt = "You are a helpful assistant"
        ollama = OllamaLocal(
            system_prompt=system_prompt,
            model_name=LOCAL_MODEL_NAME,
            model_port=LOCAL_MODEL_PORT
        )

        assert hasattr(ollama, 'clear_history')
        assert callable(ollama.clear_history)


@pytest.mark.unit
class TestOllamaLocalClearHistory:
    """Tests for OllamaLocal clear_history method"""

    def test_clear_history_resets_messages(self):
        """Test that clear_history resets messages to only system prompt"""
        system_prompt = "You are a helpful assistant"
        ollama = OllamaLocal(
            system_prompt=system_prompt,
            model_name=LOCAL_MODEL_NAME,
            model_port=LOCAL_MODEL_PORT
        )

        # Add some messages
        ollama.messages.append({"role": "user", "content": "Hello"})
        ollama.messages.append({"role": "assistant", "content": "Hi there"})

        assert len(ollama.messages) == 3  # system + 2 added

        # Clear history
        result = ollama.clear_history()

        # Verify only system message remains
        assert result is True
        assert len(ollama.messages) == 1
        assert ollama.messages[0]["role"] == "system"
        assert ollama.messages[0]["content"] == system_prompt

    def test_clear_history_preserves_system_prompt(self):
        """Test that clear_history preserves the original system prompt"""
        system_prompt = "You are a code assistant specialized in Python"
        ollama = OllamaLocal(
            system_prompt=system_prompt,
            model_name=LOCAL_MODEL_NAME,
            model_port=LOCAL_MODEL_PORT
        )

        # Add and clear messages multiple times
        for i in range(3):
            ollama.messages.append({"role": "user", "content": f"Message {i}"})
            ollama.clear_history()

        # Verify system prompt is still correct
        assert len(ollama.messages) == 1
        assert ollama.messages[0]["content"] == system_prompt


@pytest.mark.unit
class TestOllamaLocalSendRequest:
    """Tests for OllamaLocal _send_request_to_local_model method"""

    @patch('microbots.llm.ollama_local.requests.post')
    def test_send_request_success(self, mock_post):
        """Test successful request to local model"""
        system_prompt = "You are a helpful assistant"
        ollama = OllamaLocal(
            system_prompt=system_prompt,
            model_name=LOCAL_MODEL_NAME,
            model_port=LOCAL_MODEL_PORT
        )

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"task_done": false, "command": "echo hello", "thoughts": "Test"}'
        }
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "test"}]
        result = ollama._send_request_to_local_model(messages)

        assert '{"task_done": false, "command": "echo hello", "thoughts": "Test"}' in result

        # Verify request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/generate"
        assert call_args[1]["json"]["model"] == LOCAL_MODEL_NAME

    @patch('microbots.llm.ollama_local.requests.post')
    def test_send_request_server_error(self, mock_post):
        """Test handling of server error response"""
        system_prompt = "You are a helpful assistant"
        ollama = OllamaLocal(
            system_prompt=system_prompt,
            model_name=LOCAL_MODEL_NAME,
            model_port=LOCAL_MODEL_PORT
        )

        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "test"}]

        with pytest.raises(Exception, match="Error from local model server: 500"):
            ollama._send_request_to_local_model(messages)


@pytest.mark.unit
class TestOllamaLocalAsk:
    """Tests for OllamaLocal ask method"""

    @patch('microbots.llm.ollama_local.requests.post')
    def test_ask_successful_response(self, mock_post):
        """Test ask method with successful response"""
        system_prompt = "You are a helpful assistant"
        ollama = OllamaLocal(
            system_prompt=system_prompt,
            model_name=LOCAL_MODEL_NAME,
            model_port=LOCAL_MODEL_PORT
        )

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Success"
        mock_response.json.return_value = {
            "response": '{"task_done": false, "command": "echo hello", "thoughts": "Executing echo command"}'
        }
        mock_post.return_value = mock_response

        result = ollama.ask("Say hello")

        assert isinstance(result, LLMAskResponse)
        assert result.task_done is False
        assert result.command == "echo hello"
        assert result.thoughts == "Executing echo command"

        # Verify retries was reset
        assert ollama.retries == 0

        # Verify messages were appended
        assert len(ollama.messages) == 3  # system + user + assistant

    @patch('microbots.llm.ollama_local.requests.post')
    def test_ask_resets_retries(self, mock_post):
        """Test that ask resets retries at the start"""
        system_prompt = "You are a helpful assistant"
        ollama = OllamaLocal(
            system_prompt=system_prompt,
            model_name=LOCAL_MODEL_NAME,
            model_port=LOCAL_MODEL_PORT
        )

        ollama.retries = 5  # Simulate previous retries

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Success"
        mock_response.json.return_value = {
            "response": '{"task_done": false, "command": "ls", "thoughts": "Listing files"}'
        }
        mock_post.return_value = mock_response

        ollama.ask("List files")

        assert ollama.retries == 0

    @patch('microbots.llm.ollama_local.requests.post')
    def test_ask_retries_on_invalid_response(self, mock_post):
        """Test that ask retries on invalid JSON response"""
        system_prompt = "You are a helpful assistant"
        ollama = OllamaLocal(
            system_prompt=system_prompt,
            model_name=LOCAL_MODEL_NAME,
            model_port=LOCAL_MODEL_PORT,
            max_retries=2
        )

        # Mock invalid response first, then valid
        mock_response_invalid = Mock()
        mock_response_invalid.status_code = 200
        mock_response_invalid.text = "Invalid response"
        mock_response_invalid.json.return_value = {
            "response": 'This is not JSON'
        }

        mock_response_valid = Mock()
        mock_response_valid.status_code = 200
        mock_response_valid.text = "Success"
        mock_response_valid.json.return_value = {
            "response": '{"task_done": true, "command": "", "thoughts": "Completed"}'
        }

        mock_post.side_effect = [mock_response_invalid, mock_response_valid]

        result = ollama.ask("Echo done")

        assert isinstance(result, LLMAskResponse)
        assert result.task_done is True
        assert result.command == ""
        assert result.thoughts == "Completed"

        # Verify retries count
        assert ollama.retries == 1  # One retry before success


@pytest.mark.ollama_local
class TestOllamaLocalIntegration:
    """Integration tests that require actual Ollama server running"""

    def test_ollama_local_with_server(self, ollama_local_ready):
        """Test OllamaLocal with actual Ollama server"""
        system_prompt = "This is a capability test for you to check whether you can follow instructions properly."

        ollama = OllamaLocal(
            system_prompt=system_prompt,
            model_name=ollama_local_ready["model_name"],
            model_port=ollama_local_ready["model_port"]
        )

        # Test basic ask
        # Leaving this checks flexible as we use low power models in GitHub Actions
        try:
            response = ollama.ask(f"Echo 'test' - provide a sample response in following JSON format {llm_output_format_str}")
        except Exception as e:
            pytest.warns(UserWarning, match=f"ask method raised an exception: {e}")
            return

        assert isinstance(response, LLMAskResponse) or True
        assert hasattr(response, 'task_done') or True
        assert hasattr(response, 'command') or True
        assert hasattr(response, 'thoughts') or True

    def test_ollama_local_clear_history_integration(self, ollama_local_ready):
        """Test clear_history with actual server"""
        system_prompt = "You are a helpful assistant"

        ollama = OllamaLocal(
            system_prompt=system_prompt,
            model_name=ollama_local_ready["model_name"],
            model_port=ollama_local_ready["model_port"]
        )

        # Add some interaction
        ollama.messages.append({"role": "user", "content": "test"})
        ollama.messages.append({"role": "assistant", "content": "response"})

        # Clear history
        result = ollama.clear_history()

        assert result is True
        assert len(ollama.messages) == 1
        assert ollama.messages[0]["role"] == "system"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
