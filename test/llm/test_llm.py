"""
Unit tests for LLM interface and response validation
"""
import pytest
import json
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from microbots.llm.llm import LLMInterface, llmAskResponse, llm_output_format_str


class ConcreteLLM(LLMInterface):
    """Concrete implementation of LLMInterface for testing"""

    def __init__(self, max_retries=3):
        self.max_retries = max_retries
        self.retries = 0
        self.messages = []

    def ask(self, message: str) -> llmAskResponse:
        """Simple implementation for testing"""
        return llmAskResponse(task_done=False, command="test", result=None)

    def clear_history(self) -> bool:
        """Simple implementation for testing"""
        self.messages = []
        return True


class TestLlmAskResponse:
    """Tests for llmAskResponse dataclass"""

    def test_default_values(self):
        """Test that default values are set correctly"""
        response = llmAskResponse()
        assert response.task_done is False
        assert response.command == ""
        assert response.result is None

    def test_custom_values(self):
        """Test creating response with custom values"""
        response = llmAskResponse(
            task_done=True,
            command="echo 'hello'",
            result="Task completed successfully"
        )
        assert response.task_done is True
        assert response.command == "echo 'hello'"
        assert response.result == "Task completed successfully"

    def test_partial_initialization(self):
        """Test partial initialization with some defaults"""
        response = llmAskResponse(command="ls -la")
        assert response.task_done is False
        assert response.command == "ls -la"
        assert response.result is None


class TestValidateLlmResponse:
    """Tests for LLMInterface._validate_llm_response method"""

    @pytest.fixture
    def llm(self):
        """Create a concrete LLM instance for testing"""
        return ConcreteLLM(max_retries=3)

    def test_valid_response_task_not_done(self, llm):
        """Test validation of a valid response with task_done=False"""
        response = json.dumps({
            "task_done": False,
            "command": "echo 'hello world'",
            "result": None
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is True
        assert llm_response.task_done is False
        assert llm_response.command == "echo 'hello world'"
        assert llm_response.result is None
        assert llm.retries == 0

    def test_valid_response_task_done(self, llm):
        """Test validation of a valid response with task_done=True"""
        response = json.dumps({
            "task_done": True,
            "command": "",
            "result": "Task completed successfully"
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is True
        assert llm_response.task_done is True
        assert llm_response.command == ""
        assert llm_response.result == "Task completed successfully"
        assert llm.retries == 0

    def test_invalid_json(self, llm):
        """Test validation with invalid JSON"""
        response = "This is not valid JSON { invalid }"

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is False
        assert llm_response is None
        assert llm.retries == 1
        assert len(llm.messages) == 1
        assert "LLM_RES_ERROR" in llm.messages[0]["content"]
        assert "correct JSON format" in llm.messages[0]["content"]

    def test_missing_required_fields(self, llm):
        """Test validation with missing required fields"""
        response = json.dumps({
            "task_done": False,
            # Missing "command" and "result"
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is False
        assert llm_response is None
        assert llm.retries == 1
        assert len(llm.messages) == 1
        assert "missing required fields" in llm.messages[0]["content"]

    def test_task_done_not_boolean(self, llm):
        """Test validation when task_done is not a boolean"""
        response = json.dumps({
            "task_done": "yes",  # Should be boolean
            "command": "echo test",
            "result": None
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is False
        assert llm_response is None
        assert llm.retries == 1
        assert "task_done" in llm.messages[0]["content"]
        assert "boolean" in llm.messages[0]["content"]

    def test_empty_command_when_task_not_done(self, llm):
        """Test validation when command is empty but task_done is False"""
        response = json.dumps({
            "task_done": False,
            "command": "",  # Empty command
            "result": None
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is False
        assert llm_response is None
        assert llm.retries == 1
        assert "command" in llm.messages[0]["content"]
        assert "non-empty string" in llm.messages[0]["content"]

    def test_whitespace_only_command_when_task_not_done(self, llm):
        """Test validation when command is whitespace only but task_done is False"""
        response = json.dumps({
            "task_done": False,
            "command": "   ",  # Whitespace only
            "result": None
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is False
        assert llm_response is None
        assert llm.retries == 1

    def test_null_command_when_task_not_done(self, llm):
        """Test validation when command is null but task_done is False"""
        response = json.dumps({
            "task_done": False,
            "command": None,  # Null command
            "result": None
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is False
        assert llm_response is None
        assert llm.retries == 1

    def test_non_empty_command_when_task_done(self, llm):
        """Test validation when command is not empty but task_done is True"""
        response = json.dumps({
            "task_done": True,
            "command": "echo 'should not have this'",  # Should be empty
            "result": "Done"
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is False
        assert llm_response is None
        assert llm.retries == 1
        assert "command" in llm.messages[0]["content"]
        assert "empty string" in llm.messages[0]["content"]

    def test_max_retries_exceeded(self, llm):
        """Test that exception is raised when max retries is exceeded"""
        llm.retries = 3  # Set to max

        response = json.dumps({
            "task_done": False,
            "command": "",  # Invalid
            "result": None
        })

        with pytest.raises(Exception) as exc_info:
            llm._validate_llm_response(response)

        assert "Maximum retries reached" in str(exc_info.value)

    def test_retry_increments(self, llm):
        """Test that retries increment correctly on each validation failure"""
        assert llm.retries == 0

        # First invalid response
        response = json.dumps({"task_done": "invalid"})
        llm._validate_llm_response(response)
        assert llm.retries == 1

        # Second invalid response
        llm._validate_llm_response(response)
        assert llm.retries == 2

        # Third invalid response
        llm._validate_llm_response(response)
        assert llm.retries == 3

    def test_valid_response_with_result_string(self, llm):
        """Test validation with result as a string"""
        response = json.dumps({
            "task_done": True,
            "command": "",
            "result": "Analysis complete: Found 5 errors"
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is True
        assert llm_response.result == "Analysis complete: Found 5 errors"

    def test_valid_response_with_null_result(self, llm):
        """Test validation with result as null"""
        response = json.dumps({
            "task_done": False,
            "command": "ls -la",
            "result": None
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is True
        assert llm_response.result is None

    def test_command_with_special_characters(self, llm):
        """Test validation with command containing special characters"""
        response = json.dumps({
            "task_done": False,
            "command": "echo 'Hello \"World\"' | grep -i 'world'",
            "result": None
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is True
        assert llm_response.command == "echo 'Hello \"World\"' | grep -i 'world'"

    def test_extra_fields_ignored(self, llm):
        """Test that extra fields in response are ignored"""
        response = json.dumps({
            "task_done": False,
            "command": "echo test",
            "result": None,
            "extra_field": "should be ignored",
            "another_extra": 123
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is True
        assert not hasattr(llm_response, "extra_field")
        assert not hasattr(llm_response, "another_extra")

    def test_task_done_false_boolean(self, llm):
        """Test validation with task_done explicitly set to False"""
        response = json.dumps({
            "task_done": False,
            "command": "pwd",
            "result": None
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is True
        assert llm_response.task_done is False

    def test_task_done_true_boolean(self, llm):
        """Test validation with task_done explicitly set to True"""
        response = json.dumps({
            "task_done": True,
            "command": "",
            "result": "All tasks completed"
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is True
        assert llm_response.task_done is True

    def test_command_with_newlines(self, llm):
        """Test validation with multi-line command"""
        response = json.dumps({
            "task_done": False,
            "command": "for i in 1 2 3; do\n  echo $i\ndone",
            "result": None
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is True
        assert "\n" in llm_response.command

    def test_error_message_appended_to_messages(self, llm):
        """Test that error messages are appended to messages list"""
        response = json.dumps({
            "task_done": "not a boolean",
            "command": "test",
            "result": None
        })

        initial_message_count = len(llm.messages)
        llm._validate_llm_response(response)

        assert len(llm.messages) == initial_message_count + 1
        assert llm.messages[-1]["role"] == "user"
        assert "LLM_RES_ERROR" in llm.messages[-1]["content"]

    def test_multiple_validation_failures(self, llm):
        """Test multiple consecutive validation failures"""
        # First failure - invalid JSON
        llm._validate_llm_response("invalid json")
        assert llm.retries == 1

        # Second failure - missing fields
        llm._validate_llm_response(json.dumps({"task_done": False}))
        assert llm.retries == 2

        # Third failure - empty command
        llm._validate_llm_response(json.dumps({
            "task_done": False,
            "command": "",
            "result": None
        }))
        assert llm.retries == 3

        # Should have 3 error messages
        assert len(llm.messages) == 3


class TestLlmOutputFormatStr:
    """Test the output format string constant"""

    def test_format_string_contains_required_fields(self):
        """Test that the format string contains all required field names"""
        assert "task_done" in llm_output_format_str
        assert "command" in llm_output_format_str
        assert "result" in llm_output_format_str

    def test_format_string_contains_types(self):
        """Test that the format string shows the types"""
        assert "bool" in llm_output_format_str
        assert "str" in llm_output_format_str
        assert "null" in llm_output_format_str


class TestConcreteLLMImplementation:
    """Test the concrete LLM implementation used for testing"""

    def test_ask_returns_llmAskResponse(self):
        """Test that ask method returns correct type"""
        llm = ConcreteLLM()
        response = llm.ask("test message")

        assert isinstance(response, llmAskResponse)

    def test_clear_history(self):
        """Test that clear_history clears messages"""
        llm = ConcreteLLM()
        llm.messages = [{"role": "user", "content": "test"}]

        result = llm.clear_history()

        assert result is True
        assert len(llm.messages) == 0

    def test_max_retries_initialization(self):
        """Test that max_retries is set correctly"""
        llm = ConcreteLLM(max_retries=5)
        assert llm.max_retries == 5
        assert llm.retries == 0
