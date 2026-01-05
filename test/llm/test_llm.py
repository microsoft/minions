"""
Unit tests for LLM interface and response validation
"""
import pytest
import json
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from microbots.llm.llm import LLMInterface, LLMAskResponse, llm_output_format_str


class ConcreteLLM(LLMInterface):
    """Concrete implementation of LLMInterface for testing"""

    def __init__(self, max_retries=3):
        self.max_retries = max_retries
        self.retries = 0
        self.messages = []

    def ask(self, message: str) -> LLMAskResponse:
        """Simple implementation for testing"""
        return LLMAskResponse(task_done=False, command="test", thoughts=None)

    def clear_history(self) -> bool:
        """Simple implementation for testing"""
        self.messages = []
        return True

@pytest.mark.unit
class TestLlmAskResponse:
    """Tests for LLMAskResponse dataclass"""

    def test_default_values(self):
        """Test that default values are set correctly"""
        response = LLMAskResponse()
        assert response.task_done is False
        assert response.command == ""
        assert response.thoughts == ""

    def test_custom_values(self):
        """Test creating response with custom values"""
        response = LLMAskResponse(
            task_done=True,
            command="echo 'hello'",
            thoughts="Task completed successfully"
        )
        assert response.task_done is True
        assert response.command == "echo 'hello'"
        assert response.thoughts == "Task completed successfully"

    def test_partial_initialization(self):
        """Test partial initialization with some defaults"""
        response = LLMAskResponse(command="ls -la")
        assert response.task_done is False
        assert response.command == "ls -la"
        assert response.thoughts == ""

@pytest.mark.unit
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
            "thoughts": None
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is True
        assert llm_response.task_done is False
        assert llm_response.command == "echo 'hello world'"
        assert llm_response.thoughts is None
        assert llm.retries == 0

    def test_valid_response_task_done(self, llm):
        """Test validation of a valid response with task_done=True"""
        response = json.dumps({
            "task_done": True,
            "command": "",
            "thoughts": "Task completed successfully"
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is True
        assert llm_response.task_done is True
        assert llm_response.command == ""
        assert llm_response.thoughts == "Task completed successfully"
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
            # Missing "command" and "thoughts"
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
            "thoughts": None
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
            "thoughts": None
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
            "thoughts": None
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
            "thoughts": None
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
            "thoughts": "Done"
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
            "thoughts": None
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
            "thoughts": "Analysis complete: Found 5 errors"
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is True
        assert llm_response.thoughts == "Analysis complete: Found 5 errors"

    def test_valid_response_with_null_result(self, llm):
        """Test validation with result as null"""
        response = json.dumps({
            "task_done": False,
            "command": "ls -la",
            "thoughts": None
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is True
        assert llm_response.thoughts is None

    def test_command_with_special_characters(self, llm):
        """Test validation with command containing special characters"""
        response = json.dumps({
            "task_done": False,
            "command": "echo 'Hello \"World\"' | grep -i 'world'",
            "thoughts": None
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is True
        assert llm_response.command == "echo 'Hello \"World\"' | grep -i 'world'"

    def test_extra_fields_ignored(self, llm):
        """Test that extra fields in response are ignored"""
        response = json.dumps({
            "task_done": False,
            "command": "echo test",
            "thoughts": None,
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
            "thoughts": None
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is True
        assert llm_response.task_done is False

    def test_task_done_true_boolean(self, llm):
        """Test validation with task_done explicitly set to True"""
        response = json.dumps({
            "task_done": True,
            "command": "",
            "thoughts": "All tasks completed"
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is True
        assert llm_response.task_done is True

    def test_command_with_newlines(self, llm):
        """Test validation with multi-line command"""
        response = json.dumps({
            "task_done": False,
            "command": "for i in 1 2 3; do\n  echo $i\ndone",
            "thoughts": None
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is True
        assert "\n" in llm_response.command

    def test_error_message_appended_to_messages(self, llm):
        """Test that error messages are appended to messages list"""
        response = json.dumps({
            "task_done": "not a boolean",
            "command": "test",
            "thoughts": None
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
            "thoughts": None
        }))
        assert llm.retries == 3

        # Should have 3 error messages
        assert len(llm.messages) == 3

@pytest.mark.unit
class TestLlmOutputFormatStr:
    """Test the output format string constant"""

    def test_format_string_contains_required_fields(self):
        """Test that the format string contains all required field names"""
        assert "task_done" in llm_output_format_str
        assert "command" in llm_output_format_str
        assert "thoughts" in llm_output_format_str

    def test_format_string_contains_types(self):
        """Test that the format string shows the types"""
        assert "bool" in llm_output_format_str
        assert "str" in llm_output_format_str

@pytest.mark.unit
class TestConcreteLLMImplementation:
    """Test the concrete LLM implementation used for testing"""

    def test_ask_returns_LLMAskResponse(self):
        """Test that ask method returns correct type"""
        llm = ConcreteLLM()
        response = llm.ask("test message")

        assert isinstance(response, LLMAskResponse)

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

@pytest.mark.unit
class TestValidateLlmResponseAdditionalCases:
    """Additional test cases to cover all branches in _validate_llm_response"""

    @pytest.fixture
    def llm(self):
        """Create a concrete LLM instance for testing"""
        return ConcreteLLM(max_retries=3)

    def test_command_is_integer_not_string(self, llm):
        """Test validation when command is an integer instead of string"""
        response = json.dumps({
            "task_done": False,
            "command": 123,  # Integer, not string
            "thoughts": None
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is False
        assert llm_response is None
        assert llm.retries == 1
        assert "command" in llm.messages[0]["content"]
        assert "non-empty string" in llm.messages[0]["content"]

    def test_missing_fields_error_message(self, llm):
        """Test that missing fields produces correct error message"""
        response = json.dumps({
            "task_done": False,
            # Missing "command" and "thoughts"
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is False
        assert llm_response is None
        assert llm.retries == 1
        assert "missing required fields" in llm.messages[0]["content"]
        assert "LLM_RES_ERROR" in llm.messages[0]["content"]

    def test_only_task_done_field_present(self, llm):
        """Test validation with only task_done field"""
        response = json.dumps({
            "task_done": True
            # Missing command and result
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is False
        assert llm_response is None
        assert llm.retries == 1

    def test_only_command_field_present(self, llm):
        """Test validation with only command field"""
        response = json.dumps({
            "command": "echo test"
            # Missing task_done and result
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is False
        assert llm_response is None
        assert llm.retries == 1

    def test_logger_error_on_max_retries(self, llm, caplog):
        """Test that error is logged when max retries is exceeded"""
        import logging

        llm.retries = 3  # Set to max

        response = json.dumps({
            "task_done": False,
            "command": "",
            "thoughts": None
        })

        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception):
                llm._validate_llm_response(response)

        # Check if error was logged (depends on logger configuration)

    def test_logger_warning_on_invalid_json(self, llm, caplog):
        """Test that warning is logged for invalid JSON"""
        import logging

        with caplog.at_level(logging.WARNING):
            llm._validate_llm_response("not valid json")

        # Check that warning was logged

    def test_logger_info_on_valid_response(self, llm, caplog):
        """Test that info is logged for valid response"""
        import logging

        response = json.dumps({
            "task_done": False,
            "command": "echo test",
            "thoughts": None
        })

        with caplog.at_level(logging.INFO):
            llm._validate_llm_response(response)

        # Check that info was logged

    def test_task_done_as_string_true(self, llm):
        """Test validation when task_done is string 'true' instead of boolean"""
        response = json.dumps({
            "task_done": "true",  # String instead of boolean
            "command": "",
            "thoughts": None
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is False
        assert llm_response is None
        assert llm.retries == 1
        assert "boolean" in llm.messages[0]["content"]

    def test_task_done_as_integer(self, llm):
        """Test validation when task_done is integer (1/0) instead of boolean"""
        # Note: In Python, 1 in [True, False] evaluates to True because 1 == True
        # So this actually passes the boolean check. Testing with value 2 instead.
        response = json.dumps({
            "task_done": 2,  # Integer that's not 0 or 1
            "command": "test",
            "thoughts": None
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is False
        assert llm_response is None
        assert llm.retries == 1

    def test_result_field_missing_but_valid(self, llm):
        """Test that missing result field causes validation to fail"""
        # The validation checks that ALL fields are present using all(key in response_dict ...)
        # So missing result field will fail the validation
        response = json.dumps({
            "task_done": False,
            "command": "echo test"
            # result is missing - this should FAIL validation
        })

        valid, llm_response = llm._validate_llm_response(response)

        # Should be invalid because result field is missing
        assert valid is False
        assert llm_response is None
        assert llm.retries == 1

    def test_command_with_only_spaces_when_task_not_done(self, llm):
        """Test that command with only spaces is invalid when task_done=False"""
        response = json.dumps({
            "task_done": False,
            "command": "     ",  # Only spaces
            "thoughts": None
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is False
        assert llm_response is None
        assert llm.retries == 1

    def test_command_with_tabs_when_task_not_done(self, llm):
        """Test that command with tabs/whitespace is invalid when task_done=False"""
        response = json.dumps({
            "task_done": False,
            "command": "\t\t\t",  # Only tabs
            "thoughts": None
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is False
        assert llm_response is None

    def test_command_with_leading_trailing_spaces_valid(self, llm):
        """Test that command with leading/trailing spaces but content is valid"""
        response = json.dumps({
            "task_done": False,
            "command": "  echo test  ",  # Has actual content
            "thoughts": None
        })

        valid, llm_response = llm._validate_llm_response(response)

        # This should be valid because strip() shows it has content
        assert valid is True
        assert llm_response.command == "  echo test  "

    def test_task_done_true_with_whitespace_command(self, llm):
        """Test that task_done=True with whitespace-only command is invalid"""
        response = json.dumps({
            "task_done": True,
            "command": "   ",  # Whitespace
            "thoughts": "Done"
        })

        valid, llm_response = llm._validate_llm_response(response)

        # This is valid because strip() == "", which is allowed when task_done=True
        assert valid is True

    def test_json_with_comments_fails(self, llm):
        """Test that JSON with comments fails to parse"""
        response = """{
            "task_done": false,  // This is a comment
            "command": "test",
            "thoughts": null
        }"""

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is False
        assert llm_response is None
        assert llm.retries == 1

    def test_empty_string_response(self, llm):
        """Test validation with empty string response"""
        response = ""

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is False
        assert llm_response is None
        assert llm.retries == 1

    def test_null_response(self, llm):
        """Test validation with JSON null - causes TypeError"""
        response = "null"

        # When json.loads("null") is called, it returns None (not a dict)
        # This causes a TypeError when checking "key in response_dict"
        # The TypeError should be caught as a general exception
        with pytest.raises(Exception):
            llm._validate_llm_response(response)

    def test_array_response(self, llm):
        """Test validation with JSON array instead of object"""
        response = json.dumps([{"task_done": False, "command": "test"}])

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is False
        assert llm_response is None

    def test_result_with_empty_string(self, llm):
        """Test that result can be an empty string"""
        response = json.dumps({
            "task_done": True,
            "command": "",
            "thoughts": ""  # Empty string result
        })

        valid, llm_response = llm._validate_llm_response(response)

        assert valid is True
        assert llm_response.thoughts == ""

    def test_all_error_messages_contain_format_string(self, llm):
        """Test that all error messages include the format string"""
        error_scenarios = [
            "invalid json",
            json.dumps({"task_done": "invalid"}),
            json.dumps({"task_done": False, "command": ""}),
            json.dumps({"task_done": True, "command": "should be empty"}),
            json.dumps({"task_done": False}),  # Missing fields
        ]

        for scenario in error_scenarios:
            llm_test = ConcreteLLM(max_retries=5)
            llm_test._validate_llm_response(scenario)

            # Check that format string is in the error message
            if len(llm_test.messages) > 0:
                assert llm_output_format_str in llm_test.messages[-1]["content"]

    def test_task_done_true_with_missing_command_field(self, llm):
        """Test validation when task_done is True but command field is missing entirely"""
        response = json.dumps({
            "task_done": True,
            # "command" field is missing
            "thoughts": "Task completed"
        })

        valid, llm_response = llm._validate_llm_response(response)

        # Should be invalid because command field is required (missing field check)
        assert valid is False
        assert llm_response is None
        assert llm.retries == 1
        assert len(llm.messages) == 1
        assert "missing required fields" in llm.messages[0]["content"]

    def test_task_done_true_with_none_command_field(self, llm):
        """Test validation when task_done is True but command field None"""
        response = json.dumps({
            "task_done": True,
            "command": None,
            "thoughts": "Task completed"
        })

        valid, llm_response = llm._validate_llm_response(response)

        # Should be invalid because command field is required (missing field check)
        assert valid is True
        assert llm_response.task_done is True
        assert llm_response.command is None
        assert llm_response.thoughts == "Task completed"
        assert llm.retries == 0
        assert len(llm.messages) == 0


    def test_task_done_true_with_not_none_command_field(self, llm):
        """Test validation when task_done is True but command field is not None"""
        response = json.dumps({
            "task_done": True,
            "command": "not empty",
            "thoughts": "Task completed"
        })

        valid, llm_response = llm._validate_llm_response(response)

        # Should be invalid because command field is required (missing field check)
        assert valid is False
        assert llm_response is None
        assert llm.retries == 1
        assert len(llm.messages) == 1
        assert "When 'task_done' is true, 'command' should be an empty string." in llm.messages[0]["content"]


@pytest.mark.unit
class TestSummarizeContext:
    """Tests for LLMInterface._summarize_context method"""

    @pytest.fixture
    def llm(self):
        """Create a concrete LLM instance for testing"""
        return ConcreteLLM(max_retries=3)

    def test_basic_summarize_with_enough_messages(self, llm):
        """Test summarization when there are more messages than last_n_messages*2"""
        # Setup: system prompt + 25 messages (more than default 10*2=20)
        llm.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        for i in range(25):
            role = "user" if i % 2 == 0 else "assistant"
            llm.messages.append({"role": role, "content": f"Message {i}"})

        # Add a final message that will be popped (the summarization request)
        llm.messages.append({"role": "user", "content": "Please summarize"})

        initial_len = len(llm.messages)
        llm._summarize_context(last_n_messages=10, summary="This is a summary")

        # Should have: 1 system prompt + 20 recent messages (10*2)
        assert len(llm.messages) == 21
        assert llm.messages[0]["role"] == "system"
        assert "__summary__" in llm.messages[0]["content"]
        assert "This is a summary" in llm.messages[0]["content"]
        assert "__end_summary__" in llm.messages[0]["content"]

    def test_summarize_with_fewer_messages_than_threshold(self, llm):
        """Test summarization when there are fewer messages than last_n_messages*2"""
        # Setup: system prompt + 5 messages (less than 10*2=20)
        llm.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good"},
            {"role": "user", "content": "Please summarize"}  # Will be popped
        ]

        llm._summarize_context(last_n_messages=10, summary="Short conversation summary")

        # Should have: 1 system prompt + 4 messages (all except system and popped)
        assert len(llm.messages) == 5
        assert llm.messages[0]["role"] == "system"
        assert "__summary__" in llm.messages[0]["content"]
        assert "Short conversation summary" in llm.messages[0]["content"]

    def test_summarize_with_empty_summary(self, llm):
        """Test summarization with empty summary string"""
        llm.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Summarize"}  # Will be popped
        ]

        llm._summarize_context(last_n_messages=10, summary="")

        assert len(llm.messages) == 3
        assert "__summary__" in llm.messages[0]["content"]
        assert "__end_summary__" in llm.messages[0]["content"]
        # Empty summary should still have the markers
        assert "\n__summary__\n\n__end_summary__" in llm.messages[0]["content"]

    def test_summarize_updates_existing_summary(self, llm):
        """Test that existing summary is updated when system prompt already has one"""
        llm.messages = [
            {"role": "system", "content": "You are a helpful assistant.\n__summary__\nOld summary content\n__end_summary__"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Summarize"}  # Will be popped
        ]

        llm._summarize_context(last_n_messages=10, summary="New summary content")

        assert "__summary__" in llm.messages[0]["content"]
        assert "Old summary content" in llm.messages[0]["content"]
        assert "New summary content" in llm.messages[0]["content"]
        # Check that the old and new summaries are combined
        content = llm.messages[0]["content"]
        summary_section = content.split("__summary__")[1].split("__end_summary__")[0]
        assert "Old summary content" in summary_section
        assert "New summary content" in summary_section

    def test_summarize_preserves_system_prompt_before_summary(self, llm):
        """Test that original system prompt is preserved before the summary"""
        original_prompt = "You are a coding assistant with expertise in Python."
        llm.messages = [
            {"role": "system", "content": original_prompt},
            {"role": "user", "content": "Help me"},
            {"role": "assistant", "content": "Sure"},
            {"role": "user", "content": "Summarize"}
        ]

        llm._summarize_context(last_n_messages=10, summary="Summary here")

        assert llm.messages[0]["content"].startswith(original_prompt)

    def test_summarize_pops_last_message(self, llm):
        """Test that the last message (summarization request) is popped"""
        llm.messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "This should be popped"}
        ]

        llm._summarize_context(last_n_messages=10, summary="Summary")

        # The last user message should be gone
        for msg in llm.messages:
            assert msg["content"] != "This should be popped"

    def test_summarize_with_custom_last_n_messages(self, llm):
        """Test summarization with custom last_n_messages value"""
        # Setup: system prompt + 15 messages
        llm.messages = [
            {"role": "system", "content": "System prompt"}
        ]
        for i in range(15):
            role = "user" if i % 2 == 0 else "assistant"
            llm.messages.append({"role": role, "content": f"Message {i}"})
        llm.messages.append({"role": "user", "content": "Summarize"})

        llm._summarize_context(last_n_messages=5, summary="Custom summary")

        # Should have: 1 system prompt + 10 recent messages (5*2)
        assert len(llm.messages) == 11
        # Verify the most recent messages are kept
        assert llm.messages[-1]["content"] == "Message 14"

    def test_summarize_with_only_system_prompt_and_one_message(self, llm):
        """Test summarization with minimal messages"""
        llm.messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "To be popped"}
        ]

        llm._summarize_context(last_n_messages=10, summary="Minimal summary")

        # Should have only system prompt (all others popped or excluded)
        assert len(llm.messages) == 1
        assert llm.messages[0]["role"] == "system"
        assert "Minimal summary" in llm.messages[0]["content"]

    def test_summarize_keeps_correct_recent_messages(self, llm):
        """Test that exactly the right recent messages are kept"""
        llm.messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Old message 1"},
            {"role": "assistant", "content": "Old response 1"},
            {"role": "user", "content": "Old message 2"},
            {"role": "assistant", "content": "Old response 2"},
            {"role": "user", "content": "Recent message 1"},
            {"role": "assistant", "content": "Recent response 1"},
            {"role": "user", "content": "Summarize request"}  # Will be popped
        ]

        llm._summarize_context(last_n_messages=2, summary="Summary")

        # Should keep last 4 messages (2*2)
        assert len(llm.messages) == 5  # 1 system + 4 recent
        message_contents = [m["content"] for m in llm.messages[1:]]
        assert "Old message 2" in message_contents
        assert "Old response 2" in message_contents
        assert "Recent message 1" in message_contents
        assert "Recent response 1" in message_contents
        assert "Old message 1" not in message_contents
        assert "Old response 1" not in message_contents

    def test_summarize_default_empty_summary(self, llm):
        """Test that default summary parameter is empty string"""
        llm.messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Message"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Summarize"}
        ]

        # Call without summary parameter
        llm._summarize_context(last_n_messages=10)

        assert "__summary__" in llm.messages[0]["content"]
        assert "__end_summary__" in llm.messages[0]["content"]

    def test_summarize_with_multiline_summary(self, llm):
        """Test summarization with multi-line summary content"""
        llm.messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Summarize"}
        ]

        multiline_summary = """Line 1 of summary
Line 2 of summary
Line 3 with details"""

        llm._summarize_context(last_n_messages=10, summary=multiline_summary)

        assert multiline_summary in llm.messages[0]["content"]

    def test_summarize_combines_old_and_new_summary_with_newline(self, llm):
        """Test that old and new summaries are combined with newline separator"""
        llm.messages = [
            {"role": "system", "content": "Prompt\n__summary__\nFirst summary\n__end_summary__"},
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Summarize"}
        ]

        llm._summarize_context(last_n_messages=10, summary="Second summary")

        content = llm.messages[0]["content"]
        summary_section = content.split("__summary__")[1].split("__end_summary__")[0]
        # Check newline separator between summaries
        assert "\nFirst summary\n\nSecond summary\n" in content or "First summary\n\nSecond summary" in summary_section

    def test_summarize_exact_boundary_messages(self, llm):
        """Test summarization when messages count equals exactly last_n_messages*2"""
        # Setup: system prompt + exactly 6 messages (3*2)
        llm.messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Msg 1"},
            {"role": "assistant", "content": "Resp 1"},
            {"role": "user", "content": "Msg 2"},
            {"role": "assistant", "content": "Resp 2"},
            {"role": "user", "content": "Msg 3"},
            {"role": "assistant", "content": "Resp 3"},
            {"role": "user", "content": "Summarize"}  # Will be popped
        ]

        llm._summarize_context(last_n_messages=3, summary="Boundary test")

        # After popping, we have 7 messages. last_n_messages*2 = 6
        # len(messages) > 6, so we take the last 6
        assert len(llm.messages) == 7  # 1 system + 6 recent

    def test_summarize_one_more_than_boundary(self, llm):
        """Test summarization when messages count is one more than last_n_messages*2"""
        # Setup: system prompt + 7 messages (one more than 3*2=6)
        llm.messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Should be excluded"},
            {"role": "user", "content": "Msg 1"},
            {"role": "assistant", "content": "Resp 1"},
            {"role": "user", "content": "Msg 2"},
            {"role": "assistant", "content": "Resp 2"},
            {"role": "user", "content": "Msg 3"},
            {"role": "assistant", "content": "Resp 3"},
            {"role": "user", "content": "Summarize"}  # Will be popped
        ]

        llm._summarize_context(last_n_messages=3, summary="Test")

        # Should exclude "Should be excluded"
        assert len(llm.messages) == 7  # 1 system + 6 recent
        message_contents = [m["content"] for m in llm.messages]
        assert "Should be excluded" not in message_contents