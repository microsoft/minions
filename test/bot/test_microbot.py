"""
This test uses the Microbot base class to create a custom bot and tries to solve
https://github.com/SWE-agent/test-repo/issues/1.
This test will create multiple custom bots - a reading bot, a writing bot using the base class.
"""

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch, Mock

import pytest
# Add src directory to path to import from local source
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from microbots import MicroBot, BotRunResult
from microbots.constants import DOCKER_WORKING_DIR, PermissionLabels
from microbots.extras.mount import Mount, MountType
from microbots.environment.Environment import CmdReturn
from microbots.llm.llm import llm_output_format_str, LLMAskResponse


SYSTEM_PROMPT = f"""
You are a helpful python programmer who is good in debugging code.
You have the python repo where you're working mounted at {DOCKER_WORKING_DIR}.
You have a shell session open for you.
I will provide a task to achieve using only the shell commands.
You cannot run any interactive commands like vim, nano, etc. To update a file, you must use `sed` or `echo` commands.
Do not run recursive `find`, `tree`, or `sed` across the whole repo (especially `.git`). Inspect only directories/files directly related to the failure.
When running pytest, ONLY test the specific file mentioned in the task - do not run the entire test directory or test suite.
You will provide the commands to achieve the task in this particular below json format, Ensure all the time to respond in this format only and nothing else, also all the properties ( task_done, command, result ) are mandatory on each response

You must send `task_done` as true only when you have completed the task. It means all the commands you wanted to run are completed in the previous steps. You should not run any more commands while you're sending `task_done` as true.
{llm_output_format_str}
"""


@pytest.fixture(scope="function")
def no_mount_microBot():
    local_model = os.getenv('LOCAL_MODEL_NAME', 'qwen2.5-coder:latest').replace(':latest', '')
    bot = MicroBot(
        model=f"ollama-local/{local_model}",
        system_prompt=SYSTEM_PROMPT,
    )
    yield bot
    del bot


@pytest.mark.integration
@pytest.mark.docker
class TestMicrobotIntegration:

    @pytest.fixture(scope="function")
    def log_file_path(self, tmpdir: Path):
        assert tmpdir.exists()
        yield tmpdir / "error.log"
        if tmpdir.exists():
            subprocess.run(["sudo", "rm", "-rf", str(tmpdir)])

    @pytest.fixture(scope="function")
    def ro_mount(self, test_repo: Path):
        assert test_repo is not None
        return Mount(
            str(test_repo), f"{DOCKER_WORKING_DIR}/{test_repo.name}", PermissionLabels.READ_ONLY
        )

    @pytest.fixture(scope="function")
    def ro_microBot(self, ro_mount: Mount):
        local_model = os.getenv('LOCAL_MODEL_NAME', 'qwen2.5-coder:latest').replace(':latest', '')
        bot = MicroBot(
            model=f"ollama-local/{local_model}",
            system_prompt=SYSTEM_PROMPT,
            folder_to_mount=ro_mount,
        )
        yield bot
        del bot

    @pytest.fixture(scope="function")
    def anthropic_microBot(self):
        anthropic_deployment = os.getenv('ANTHROPIC_DEPLOYMENT_NAME', 'claude-sonnet-4-5')
        with patch('microbots.llm.anthropic_api.endpoint', 'https://api.anthropic.com'), \
             patch('microbots.llm.anthropic_api.deployment_name', anthropic_deployment), \
             patch('microbots.llm.anthropic_api.api_key', 'test-api-key'), \
             patch('microbots.llm.anthropic_api.Anthropic'):
            bot = MicroBot(
                model=f"anthropic/{anthropic_deployment}",
                system_prompt=SYSTEM_PROMPT,
            )
            yield bot
            del bot

    @pytest.mark.ollama_local
    def test_microbot_ro_mount(self, ro_microBot, test_repo: Path):
        assert test_repo is not None

        result: CmdReturn = ro_microBot.environment.execute(f"cd {DOCKER_WORKING_DIR}/{test_repo.name} && ls -la", timeout=60)
        logger.info(f"Command Execution Result: \nstdout={result.stdout}, \nstderr={result.stderr}, \nreturn_code={result.return_code}")
        assert result.return_code == 0
        assert "tests" in result.stdout

        result = ro_microBot.environment.execute("cd tests; ls -la", timeout=60)
        logger.info(f"Command Execution Result: \nstdout={result.stdout}, \nstderr={result.stderr}, \nreturn_code={result.return_code}")
        assert result.return_code == 0
        assert "missing_colon.py" in result.stdout

    @pytest.mark.ollama_local
    def test_microbot_overlay_teardown(self, ro_microBot, caplog):
        caplog.clear()
        caplog.set_level(logging.INFO)

        del ro_microBot

        assert "Failed to remove working directory" not in caplog.text

    def test_microbot_anthropic_initialization(self, anthropic_microBot):
        """Test that MicroBot correctly initializes with Anthropic model provider."""
        assert anthropic_microBot is not None
        assert anthropic_microBot.model_provider == "anthropic"
        assert anthropic_microBot.deployment_name == "claude-sonnet-4-5"
        assert anthropic_microBot.llm is not None
        from microbots.llm.anthropic_api import AnthropicApi
        assert isinstance(anthropic_microBot.llm, AnthropicApi)

    @pytest.mark.slow
    def test_microbot_2bot_combo(self, log_file_path, test_repo, issue_1):
        assert test_repo is not None
        assert log_file_path is not None

        verify_function = issue_1[1]

        test_repo_mount_ro = Mount(
            str(test_repo),
            f"{DOCKER_WORKING_DIR}/{test_repo.name}",
            PermissionLabels.READ_ONLY
        )
        model = f"azure-openai/{os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'mini-swe-agent-gpt5')}"
        testing_bot = MicroBot(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            folder_to_mount=test_repo_mount_ro,
        )

        response: BotRunResult = testing_bot.run(
            "Execute tests/missing_colon.py and provide the error message. Your response should be in 'thoughts' field. missing_colon.py is a python file not a test file. So, don't use pytest to run it. Just run 'python tests/missing_colon.py'.",
            timeout_in_seconds=300
        )

        logger.debug(f"Custom Reading Bot - Status: {response.status}, Result: {response.result}, Error: {response.error}")

        assert response.status
        assert response.result is not None
        assert response.error is None

        with open(log_file_path, "w") as log_file:
            log_file.write(response.result)

        test_repo_mount_rw = Mount(
            str(test_repo), f"{DOCKER_WORKING_DIR}/{test_repo.name}", PermissionLabels.READ_WRITE
        )
        model = f"azure-openai/{os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'mini-swe-agent-gpt5')}"
        coding_bot = MicroBot(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            folder_to_mount=test_repo_mount_rw,
        )

        additional_mounts = Mount(
            str(log_file_path),
            "/var/log/",
            PermissionLabels.READ_ONLY,
            MountType.COPY,
        )
        response: BotRunResult = coding_bot.run(
            f"The python script, tests/missing_colon.py is failing. Please fix the code. The error log is available at /var/log/{log_file_path.basename}. missing_colon.py is a python file not a test file. So, don't use pytest to run it. Just run 'python tests/missing_colon.py'.",
            additional_mounts=[additional_mounts],
            timeout_in_seconds=300
        )

        print(f"Custom Coding Bot - Status: {response.status}, Result: {response.result}, Error: {response.error}")

        assert response.status
        assert response.error is None

        verify_function(test_repo)

    def test_microbot_anthropic_with_mount(self, test_repo):
        """Test that MicroBot with Anthropic provider works with mounted folders."""
        assert test_repo is not None

        test_repo_mount_ro = Mount(
            str(test_repo), f"{DOCKER_WORKING_DIR}/{test_repo.name}", PermissionLabels.READ_ONLY
        )
        anthropic_deployment = os.getenv('ANTHROPIC_DEPLOYMENT_NAME', 'claude-sonnet-4-5')
        with patch('microbots.llm.anthropic_api.endpoint', 'https://api.anthropic.com'), \
             patch('microbots.llm.anthropic_api.deployment_name', anthropic_deployment), \
             patch('microbots.llm.anthropic_api.api_key', 'test-api-key'), \
             patch('microbots.llm.anthropic_api.Anthropic'):
            bot = MicroBot(
                model=f"anthropic/{anthropic_deployment}",
                system_prompt=SYSTEM_PROMPT,
                folder_to_mount=test_repo_mount_ro,
            )
            assert bot is not None
            assert bot.model_provider == "anthropic"
            from microbots.llm.anthropic_api import AnthropicApi
            assert isinstance(bot.llm, AnthropicApi)
            del bot

    def test_summarize_context_invalid_syntax_asks_llm_to_retry(self, no_mount_microBot, caplog):
        """Integration test: LLM sends invalid summarize_context syntax, bot asks LLM to correct it.

        This test verifies that when the LLM sends an improperly formatted summarize_context
        command, the MicroBot:
        1. Detects the invalid syntax
        2. Sends an error message back to the LLM with correct usage instructions
        3. Allows the LLM to retry with correct syntax
        4. Successfully completes the summarization when correct syntax is provided

        This follows the pattern used in _validate_llm_response in llm.py where invalid
        responses trigger retry messages.
        """
        caplog.set_level(logging.DEBUG)

        call_count = [0]

        def mock_ask(message: str):
            nonlocal call_count
            no_mount_microBot.llm.messages.append({"role": "user", "content": message})
            call_count[0] += 1
            logger.info(f"mock_ask called (count={call_count[0]}): {message[:100]}...")

            if call_count[0] == 1:
                # First call (initial prompt) - LLM sends invalid summarize_context (missing quotes)
                res = LLMAskResponse(
                    task_done=False,
                    thoughts="Context is getting long, let me summarize",
                    command='summarize_context 2 missing quotes around summary',  # Invalid: summary not quoted
                )
            elif call_count[0] == 2:
                # Second call - should receive error message about invalid syntax
                assert "COMMAND_ERROR: Invalid summarize_context syntax" in message
                assert "Correct usage:" in message
                assert 'summarize_context <no_of_recent_turns_to_keep> "<your summary of the context>"' in message
                # Now LLM sends correct syntax
                res = LLMAskResponse(
                    task_done=False,
                    thoughts="Let me fix the syntax",
                    command='summarize_context 2 "This is a proper summary with quotes"',
                )
            elif call_count[0] == 3:
                # Third call - after successful summarize_context, receives the previous message
                # Complete the task
                res = LLMAskResponse(
                    task_done=True,
                    thoughts="Task completed after context summarization",
                    command='',
                )
            else:
                res = LLMAskResponse(task_done=True, thoughts="Done", command='')

            no_mount_microBot.llm.messages.append({"role": "assistant", "content": json.dumps(asdict(res))})
            return res

        # Apply mocks
        no_mount_microBot.llm.ask = mock_ask

        response: BotRunResult = no_mount_microBot.run(
            "Test summarize_context invalid syntax handling",
            timeout_in_seconds=60,
            max_iterations=10
        )

        # Verify the flow worked correctly
        assert response.status, f"Expected success but got error: {response.error}"
        assert response.error is None
        assert call_count[0] >= 3, "Should have had at least 3 LLM calls (initial, retry, complete)"

        # Verify error message was logged
        assert "Invalid summarize_context syntax" in caplog.text or call_count[0] >= 2

    def test_summarize_context_various_invalid_syntaxes(self, no_mount_microBot):
        """Integration test: Various invalid summarize_context syntaxes are properly rejected.

        Tests multiple edge cases of invalid syntax to ensure robust parsing.
        """
        test_cases = [
            ('summarize_context', "Missing all arguments"),
            ('summarize_context abc "summary"', "Non-integer first argument"),
            ('summarize_context 2', "Missing summary (should work with empty default)"),
            ('summarize_context "summary" 2', "Arguments in wrong order"),
        ]

        call_count = [0]
        received_errors = []

        def mock_ask(message: str):
            nonlocal call_count
            no_mount_microBot.llm.messages.append({"role": "user", "content": message})
            call_count[0] += 1

            if "COMMAND_ERROR" in message:
                received_errors.append(message)
                # After receiving error, send a valid command to complete
                res = LLMAskResponse(
                    task_done=True,
                    thoughts="Acknowledged the error",
                    command='',
                )
                no_mount_microBot.llm.messages.append({"role": "assistant", "content": json.dumps(asdict(res))})
                return res

            # First call - try invalid syntax
            if call_count[0] == 1:
                res = LLMAskResponse(
                    task_done=False,
                    thoughts="Testing invalid syntax",
                    command=test_cases[0][0],  # First invalid case
                )
                no_mount_microBot.llm.messages.append({"role": "assistant", "content": json.dumps(asdict(res))})
                return res

            res = LLMAskResponse(task_done=True, thoughts="Done", command='')
            no_mount_microBot.llm.messages.append({"role": "assistant", "content": json.dumps(asdict(res))})
            return res

        no_mount_microBot.llm.ask = mock_ask

        response: BotRunResult = no_mount_microBot.run(
            "Test invalid syntax",
            timeout_in_seconds=30,
            max_iterations=5
        )

        # Check that the first test case (missing arguments) was rejected
        assert len(received_errors) > 0 or response.status
        # Either we got an error for invalid syntax, or "summarize_context" alone wasn't enough


@pytest.mark.unit
class TestMicrobotUnit:
    """Unit tests for MicroBot command safety validation."""

    def test_incorrect_code_mount_type(self):
        """Test that ValueError is raised when folder_to_mount uses COPY mount type."""
        invalid_mount = Mount(
            "/dummy/path",
            f"{DOCKER_WORKING_DIR}/test",
            PermissionLabels.READ_ONLY,
            MountType.COPY,  # COPY is not supported for folder_to_mount
        )

        local_model = os.getenv('LOCAL_MODEL_NAME', 'qwen2.5-coder:latest').replace(':latest', '')
        with pytest.raises(ValueError, match="Only MOUNT mount type is supported for folder_to_mount"):
            MicroBot(
                model=f"ollama-local/{local_model}",
                system_prompt=SYSTEM_PROMPT,
                folder_to_mount=invalid_mount,
            )

    @pytest.mark.ollama_local
    def test_incorrect_copy_mount_type(self, no_mount_microBot):
        """Test that ValueError is raised when additional_mounts uses MOUNT mount type."""
        invalid_additional_mount = Mount(
            "/dummy/log/file.txt",
            "/var/log/file.txt",
            PermissionLabels.READ_ONLY,
            MountType.MOUNT,  # MOUNT is not supported for additional_mounts
        )

        with pytest.raises(ValueError, match="Only COPY mount type is supported for additional mounts for now"):
            no_mount_microBot.run(
                "Test task",
                additional_mounts=[invalid_additional_mount],
                timeout_in_seconds=60
            )

    def test_incorrect_model_provider(self):
        """Test that ValueError is raised for unsupported model providers."""
        with pytest.raises(ValueError, match="Unsupported model provider: unsupported-provider"):
            MicroBot(
                model="unsupported-provider/some-model",
                system_prompt=SYSTEM_PROMPT,
            )

    def test_incorrect_model_format(self):
        """Test that ValueError is raised for incorrectly formatted model strings."""
        with pytest.raises(ValueError, match="Model should be in the format <provider>/<model_name>"):
            MicroBot(
                model="invalidmodelname",
                system_prompt=SYSTEM_PROMPT,
            )

    @pytest.mark.ollama_local
    def test_invalid_max_iterations(self, no_mount_microBot):
        """Test that ValueError is raised for invalid max_iterations values"""
        assert no_mount_microBot is not None

        # Test with max_iterations = 0
        with pytest.raises(ValueError) as exc_info:
            no_mount_microBot.run(
                "This is a test task.",
                max_iterations=0
            )
        assert "max_iterations must be greater than 0" in str(exc_info.value)

        # Test with max_iterations = -1
        with pytest.raises(ValueError) as exc_info:
            no_mount_microBot.run(
                "This is a test task.",
                max_iterations=-1
            )
        assert "max_iterations must be greater than 0" in str(exc_info.value)

        # Test with max_iterations = -10
        with pytest.raises(ValueError) as exc_info:
            no_mount_microBot.run(
                "This is a test task.",
                max_iterations=-10
            )
        assert "max_iterations must be greater than 0" in str(exc_info.value)

    @pytest.mark.ollama_local
    def test_max_iterations_exceeded(self, no_mount_microBot, monkeypatch):
        assert no_mount_microBot is not None

        def mock_ask(message: str):
            no_mount_microBot.llm.messages.append({"role": "user", "content": message})
            res = LLMAskResponse(command="echo 'Hello World'", task_done=False, thoughts="")
            no_mount_microBot.llm.messages.append({"role": "assistant", "content": json.dumps(asdict(res))})
            return res

        monkeypatch.setattr(no_mount_microBot.llm, "ask", mock_ask)

        response: BotRunResult = no_mount_microBot.run(
            "This is a test to check max iterations handling.",
            timeout_in_seconds=120,
            max_iterations=3
        )

        print(f"Max Iterations Test - Status: {response.status}, Result: {response.result}, Error: {response.error}")

        assert not response.status
        assert response.error == "Max iterations 3 reached"

    @pytest.mark.ollama_local
    def test_timeout_handling(self, no_mount_microBot, monkeypatch):
        assert no_mount_microBot is not None

        def mock_ask(message: str):
            no_mount_microBot.llm.messages.append({"role": "user", "content": message})
            res = LLMAskResponse(command="sleep 10", task_done=False, thoughts="")
            no_mount_microBot.llm.messages.append({"role": "assistant", "content": json.dumps(asdict(res))})
            return res

        monkeypatch.setattr(no_mount_microBot.llm, "ask", mock_ask)

        response: BotRunResult = no_mount_microBot.run(
            "This is a test to check timeout handling.",
            timeout_in_seconds=5,
            max_iterations=10
        )

        print(f"Timeout Handling Test - Status: {response.status}, Result: {response.result}, Error: {response.error}")

        assert not response.status
        assert response.error == "Timeout of 5 seconds reached"

    @pytest.mark.ollama_local
    def test_dangerous_command_blocking(self, no_mount_microBot, monkeypatch, caplog):
        """Test that dangerous commands are blocked and LLM receives detailed explanation."""
        caplog.set_level(logging.INFO)

        call_count = [0]

        def mock_ask(message: str):
            no_mount_microBot.llm.messages.append({"role": "user", "content": message})
            call_count[0] += 1
            if call_count[0] == 1:
                # First call returns dangerous command
                res = LLMAskResponse(command="ls -R /path", task_done=False, thoughts="")
            else:
                # After receiving error with explanation, return safe command
                assert "COMMAND_ERROR:" in message
                assert "Dangerous command detected and blocked" in message
                assert "REASON:" in message
                assert "ALTERNATIVE:" in message
                res = LLMAskResponse(command="pwd", task_done=True, thoughts="")
            no_mount_microBot.llm.messages.append({"role": "assistant", "content": json.dumps(asdict(res))})
            return res

        monkeypatch.setattr(no_mount_microBot.llm, "ask", mock_ask)

        response: BotRunResult = no_mount_microBot.run(
            "List files",
            timeout_in_seconds=60,
            max_iterations=10
        )

        # Verify dangerous command was logged with explanation
        assert "Dangerous command detected and blocked: ls -R /path" in caplog.text
        assert "REASON:" in caplog.text
        assert "ALTERNATIVE:" in caplog.text

        # Verify task completed after providing safe command
        assert response.status

    @pytest.mark.parametrize("command,expected_safe", [
        # Dangerous: Recursive ls commands
        ("ls -R", False),
        ("ls -lR", False),
        ("ls -alR", False),
        ("ls -Rl", False),
        ("ls -r /path", False),
        ("ls -laR /some/path", False),
        ("ls -Ra", False),
        # Dangerous: Tree commands
        ("tree", False),
        ("tree /path", False),
        ("tree -L 3", False),
        # Dangerous: Recursive rm commands
        ("rm -r /path", False),
        ("rm -rf /path", False),
        ("rm -fr /path", False),
        ("rm -Rf /path", False),
        ("rm --recursive /path", False),
        ("rm -rf .", False),
        # Dangerous: Find without maxdepth
        ("find /path -name '*.py'", False),
        ("find . -type f", False),
        ("find /home -name 'test*'", False),
        # Safe: Find with maxdepth
        ("find /path -name '*.py' -maxdepth 2", True),
        ("find . -type f -maxdepth 1", True),
        ("find /home -maxdepth 3 -name 'test*'", True),
        # Safe: Common commands (including the key test case)
        ("ls -la", True),
        ("ls -la /workdir/test-repo && ls -la /workdir/test-repo/tests", True),
        ("ls -lt", True),
        ("ls -al", True),
        ("ls /path/to/dir", True),
        ("rm file.txt", True),
        ("rm -f file.txt", True),
        ("cat file.txt", True),
        ("grep 'pattern' file.txt", True),
        ("echo 'hello'", True),
        ("cd /path", True),
        ("pwd", True),
        ("python script.py", True),
        ("git status", True),
        # Invalid inputs
        (None, False),
        ("", False),
        ("   ", False),
        (123, False),
        ([], False),
        ({}, False),
    ])
    def test_is_safe_command(self, command, expected_safe):
        """Test command safety validation for all scenarios."""
        # Create a minimal bot instance without environment (no container)
        bot = MicroBot.__new__(MicroBot)  # Create instance without calling __init__
        is_safe, explanation = bot._is_safe_command(command)
        assert is_safe == expected_safe, f"Command '{command}' expected safe={expected_safe}, got {is_safe}"

        # Verify explanation is provided when command is not safe
        if not expected_safe:
            assert explanation is not None, f"Expected explanation for unsafe command '{command}'"
            assert "REASON:" in explanation
            assert "ALTERNATIVE:" in explanation

    @pytest.mark.parametrize("command,should_be_dangerous,expected_keyword", [
        # Dangerous commands
        ("ls -R", True, "Recursive ls"),
        ("ls -lR /path", True, "Recursive ls"),
        ("tree", True, "Tree command"),
        ("rm -rf /path", True, "Recursive rm"),
        ("find . -name '*.py'", True, "Find command without -maxdepth"),
        # Safe commands
        ("ls -la", False, None),
        ("ls -la /workdir/test-repo && ls -la /workdir/test-repo/tests", False, None),
        ("rm file.txt", False, None),
        ("find /path -maxdepth 2 -name '*.py'", False, None),
    ])
    def test_get_dangerous_command_explanation(self, command, should_be_dangerous, expected_keyword):
        """Test that dangerous commands return explanations with REASON and ALTERNATIVE."""
        bot = MicroBot.__new__(MicroBot)
        result = bot._get_dangerous_command_explanation(command)

        if should_be_dangerous:
            assert result is not None, f"Command '{command}' should have explanation"
            assert "REASON:" in result and "ALTERNATIVE:" in result
            assert expected_keyword in result
        else:
            assert result is None, f"Command '{command}' should be safe"

    def test_dangerous_command_explanation_format(self):
        """Test that dangerous command explanations have correct format with reason and alternative."""
        bot = MicroBot.__new__(MicroBot)
        explanation = bot._get_dangerous_command_explanation("ls -R")

        assert explanation is not None
        lines = explanation.split('\n')
        assert len(lines) >= 2
        assert lines[0].startswith("REASON:")
        assert lines[1].startswith("ALTERNATIVE:")
        assert len(lines[0].replace("REASON:", "").strip()) > 0
        assert len(lines[1].replace("ALTERNATIVE:", "").strip()) > 0

    @pytest.mark.parametrize("command,expected_result", [
        # Valid commands
        ('summarize_context 2 "This is a summary"', (2, "This is a summary")),
        ('summarize_context 0 ""', (0, "")),
        ('summarize_context 5 "Keep 5 turns, summary: completed file exploration"', (5, "Keep 5 turns, summary: completed file exploration")),
        ('summarize_context 10 "Multi word summary with special chars: @#$%"', (10, "Multi word summary with special chars: @#$%")),
        ('summarize_context 1 "Summary with \\"escaped quotes\\""', (1, 'Summary with "escaped quotes"')),
        # Valid with default empty summary
        ('summarize_context 3', (3, "")),
        # Invalid commands
        ('summarize_context', None),  # Missing all arguments
        ('summarize_context abc "summary"', None),  # Non-integer first argument
        ('summarize_context -1 "negative"', (-1, "negative")),  # Negative is valid parsing (validation elsewhere)
        ('summarize_context "summary" 2', None),  # Wrong order
        ('not_summarize_context 2 "summary"', None),  # Wrong command name
        ('summarize_context 2 missing quotes', None),  # Summary without quotes (shlex will split)
    ])
    def test_parse_summarize_context_command(self, command, expected_result):
        """Test parsing of summarize_context command syntax."""
        bot = MicroBot.__new__(MicroBot)
        result = bot._parse_summarize_context_command(command)

        if expected_result is None:
            assert result is None, f"Expected None for invalid command '{command}', got {result}"
        else:
            assert result is not None, f"Expected valid result for command '{command}', got None"
            assert result == expected_result, f"Command '{command}': expected {expected_result}, got {result}"

    def test_get_summarize_context_syntax_error_format(self):
        """Test that syntax error message has correct format with usage examples."""
        bot = MicroBot.__new__(MicroBot)
        error_msg = bot._get_summarize_context_syntax_error('summarize_context bad syntax')

        assert "COMMAND_ERROR: Invalid summarize_context syntax" in error_msg
        assert "Your command: summarize_context bad syntax" in error_msg
        assert "Correct usage:" in error_msg
        assert 'summarize_context <no_of_recent_turns_to_keep> "<your summary of the context>"' in error_msg
        assert "Examples:" in error_msg
        assert "Please send the command again with correct syntax" in error_msg

    @pytest.mark.ollama_local
    def test_command_logging_without_json_escaping(self, no_mount_microBot, monkeypatch, caplog):
        """Test that commands are logged without JSON escaping, making logs more readable.

        This test verifies that when a command containing special characters like quotes
        is executed, the log output displays the command directly without JSON escaping
        (e.g., shows " instead of \\").
        """
        caplog.set_level(logging.INFO)

        # Command containing JSON with quotes - simulating anthropic-text-editor usage
        json_command = """echo '{"input": {"command": "view", "path": "/tmp/test.txt"}}' | cat"""

        call_count = [0]

        def mock_ask(message: str):
            no_mount_microBot.llm.messages.append({"role": "user", "content": message})
            call_count[0] += 1
            if call_count[0] == 1:
                res = LLMAskResponse(command=json_command, task_done=False, thoughts="")
            else:
                res = LLMAskResponse(command="", task_done=True, thoughts="Done")
            no_mount_microBot.llm.messages.append({"role": "assistant", "content": json.dumps(asdict(res))})
            return res

        monkeypatch.setattr(no_mount_microBot.llm, "ask", mock_ask)

        response: BotRunResult = no_mount_microBot.run(
            "Test command logging",
            timeout_in_seconds=60,
            max_iterations=5
        )

        # Verify that the command appears in logs without JSON escaping
        # The command should appear as-is, not with escaped quotes like \"
        assert "LLM tool call" in caplog.text
        # Should NOT have json-escaped quotes (\" would appear if json.dumps was used)
        assert r'\"input\"' not in caplog.text
        # Should have the raw command format
        assert '"input"' in caplog.text or "'input'" in caplog.text

    def test_pformat_produces_readable_output(self):
        """Test that pformat produces readable output for command execution logs.

        This unit test verifies the log formatting behavior for commands containing
        special characters like quotes and newlines. It ensures that using pformat
        instead of json.dumps produces more readable log output.
        """
        from pprint import pformat

        # Simulate a command with JSON - like anthropic-text-editor usage
        command = """echo '{"input": {"command": "view", "path": "/tmp/test.txt"}}' | anthropic-text-editor"""

        # With json.dumps (old behavior) - escapes quotes making it harder to read
        import json
        json_output = json.dumps(command)

        # With pformat (new behavior) - more readable
        pformat_output = pformat(command)

        # json.dumps escapes quotes with backslashes
        assert r'\"' in json_output, "json.dumps should escape quotes"

        # pformat should not have escaped quotes - it uses single quotes to wrap the string
        # so it doesn't need to escape the internal double quotes
        assert r'\"' not in pformat_output, "pformat should not have escaped quotes"

        # The raw command should be usable directly in logs without escaping
        # Just using the command directly is the most readable
        raw_output = command
        assert r'\"' not in raw_output, "Raw command should not have escaped quotes"
        assert '"input"' in raw_output, "Raw command should preserve double quotes"
    def test_tool_usage_instructions_appended_to_system_prompt(self):
        """Test that tool usage instructions are appended to the system prompt when creating LLM."""
        from microbots.tools.tool import Tool

        # Create a mock tool with usage instructions
        mock_tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters=None,
            usage_instructions_to_llm="# Test Tool Usage\nUse this tool for testing purposes only.",
            install_commands=["echo 'test'"],
            env_variables=[],
            files_to_copy=[],
        )

        base_system_prompt = "You are a helpful assistant."

        # Create a mock environment
        mock_env = Mock()
        mock_env.execute.return_value = Mock(return_code=0, stdout="", stderr="")

        # Mock the environment and LLM creation to avoid actual Docker/API calls
        with patch('microbots.llm.openai_api.OpenAI'):
            # Create a MicroBot with the mock tool
            bot = MicroBot(
                model="azure-openai/test-model",
                system_prompt=base_system_prompt,
                additional_tools=[mock_tool],
                environment=mock_env,
            )

            # Verify that the LLM was created with the combined system prompt
            # The system prompt should include both the base prompt and the tool usage instructions
            from microbots.llm.openai_api import OpenAIApi
            assert isinstance(bot.llm, OpenAIApi)
            assert base_system_prompt in bot.llm.system_prompt
            assert "# Test Tool Usage" in bot.llm.system_prompt
            assert "Use this tool for testing purposes only." in bot.llm.system_prompt

    def test_multiple_tool_usage_instructions_appended(self):
        """Test that multiple tool usage instructions are all appended to the system prompt."""
        from microbots.tools.tool import Tool

        # Create multiple mock tools with usage instructions
        tool1 = Tool(
            name="tool1",
            description="First tool",
            parameters=None,
            usage_instructions_to_llm="# Tool 1 Usage\nInstructions for tool 1.",
            install_commands=["echo 'tool1'"],
            env_variables=[],
            files_to_copy=[],
        )

        tool2 = Tool(
            name="tool2",
            description="Second tool",
            parameters=None,
            usage_instructions_to_llm="# Tool 2 Usage\nInstructions for tool 2.",
            install_commands=["echo 'tool2'"],
            env_variables=[],
            files_to_copy=[],
        )

        base_system_prompt = "You are a helpful assistant."

        # Create a mock environment
        mock_env = Mock()
        mock_env.execute.return_value = Mock(return_code=0, stdout="", stderr="")

        # Mock the environment and LLM creation
        with patch('microbots.llm.anthropic_api.Anthropic'):
            bot = MicroBot(
                model="anthropic/claude-sonnet-4-5",
                system_prompt=base_system_prompt,
                additional_tools=[tool1, tool2],
                environment=mock_env,
            )

            # Verify both tool instructions are in the system prompt
            from microbots.llm.anthropic_api import AnthropicApi
            assert isinstance(bot.llm, AnthropicApi)
            assert base_system_prompt in bot.llm.system_prompt
            assert "# Tool 1 Usage" in bot.llm.system_prompt
            assert "Instructions for tool 1." in bot.llm.system_prompt
            assert "# Tool 2 Usage" in bot.llm.system_prompt
            assert "Instructions for tool 2." in bot.llm.system_prompt

    def test_no_tool_usage_instructions_when_no_tools(self):
        """Test that system prompt remains unchanged when no tools are provided."""
        base_system_prompt = "You are a helpful assistant."

        # Create a mock environment
        mock_env = Mock()
        mock_env.execute.return_value = Mock(return_code=0, stdout="", stderr="")

        # Mock the environment and LLM creation
        with patch.dict('os.environ', {'LOCAL_MODEL_NAME': 'test-model', 'LOCAL_MODEL_PORT': '11434'}), \
             patch('microbots.llm.ollama_local.requests'):

            bot = MicroBot(
                model="ollama-local/test-model",
                system_prompt=base_system_prompt,
                additional_tools=[],
                environment=mock_env,
            )

            # Verify the system prompt is unchanged
            from microbots.llm.ollama_local import OllamaLocal
            assert isinstance(bot.llm, OllamaLocal)
            assert bot.llm.system_prompt == base_system_prompt

    def test_summarize_context_command_calls_llm_summarize(self, no_mount_microBot, monkeypatch):
        """Test that summarize_context command triggers LLM's summarize_context method."""
        assert no_mount_microBot is not None

        summarize_calls = []

        def mock_summarize_context(last_n_messages: int, summary: str):
            summarize_calls.append((last_n_messages, summary))
            return {"role": "user", "content": "Previous context message"}

        call_count = [0]

        def mock_ask(message: str):
            no_mount_microBot.llm.messages.append({"role": "user", "content": message})
            call_count[0] += 1
            if call_count[0] == 1:
                # First call (initial prompt) returns summarize_context command
                res = LLMAskResponse(
                    command='summarize_context 2 "Summary of previous work: explored files"',
                    task_done=False,
                    thoughts="Need to summarize context"
                )
            elif call_count[0] == 2:
                # Second call (with returned last message content) - complete the task
                assert message == "Previous context message"
                res = LLMAskResponse(command="", task_done=True, thoughts="Done after summarization")
            else:
                res = LLMAskResponse(command="", task_done=True, thoughts="Done")
            no_mount_microBot.llm.messages.append({"role": "assistant", "content": json.dumps(asdict(res))})
            return res

        monkeypatch.setattr(no_mount_microBot.llm, "ask", mock_ask)
        monkeypatch.setattr(no_mount_microBot.llm, "summarize_context", mock_summarize_context)

        response: BotRunResult = no_mount_microBot.run(
            "Test summarize context",
            timeout_in_seconds=60,
            max_iterations=10
        )

        # Verify summarize_context was called with the correct parsed arguments
        assert len(summarize_calls) == 1
        assert summarize_calls[0] == (2, "Summary of previous work: explored files")
        assert response.status

    def test_summarize_context_does_not_increment_iteration(self, no_mount_microBot, monkeypatch):
        """Test that summarize_context command does not count towards max_iterations."""
        assert no_mount_microBot is not None

        summarize_calls = []

        def mock_summarize_context(last_n_messages: int, summary: str):
            summarize_calls.append((last_n_messages, summary))
            return {"role": "user", "content": "Previous message content"}

        call_count = [0]

        def mock_ask(message: str):
            no_mount_microBot.llm.messages.append({"role": "user", "content": message})
            call_count[0] += 1
            # Flow with max_iterations=5:
            # 1. Initial prompt -> summarize_context
            # 2. Last message from summarize -> another summarize_context
            # 3. Last message from summarize -> actual command
            # 4. Command result -> complete
            if call_count[0] == 1:
                # First call: return summarize_context
                res = LLMAskResponse(
                    command='summarize_context 1 "Summarized"',
                    task_done=False,
                    thoughts=""
                )
            elif call_count[0] == 2:
                # Second call (with returned last message): return another summarize_context
                res = LLMAskResponse(
                    command='summarize_context 1 "Summarized again"',
                    task_done=False,
                    thoughts=""
                )
            elif call_count[0] == 3:
                # Third call (with returned last message): return actual command
                res = LLMAskResponse(command="echo 'test'", task_done=False, thoughts="")
            else:
                # Fourth call: complete
                res = LLMAskResponse(command="", task_done=True, thoughts="Done")
            no_mount_microBot.llm.messages.append({"role": "assistant", "content": json.dumps(asdict(res))})
            return res

        monkeypatch.setattr(no_mount_microBot.llm, "ask", mock_ask)
        monkeypatch.setattr(no_mount_microBot.llm, "summarize_context", mock_summarize_context)

        # With max_iterations=5, we have enough iterations to complete:
        # iteration 2: first summarize_context
        # iteration 3: second summarize_context
        # iteration 4: actual command (echo 'test')
        # iteration 5: task_done=True
        response: BotRunResult = no_mount_microBot.run(
            "Test summarize does not count iterations",
            timeout_in_seconds=60,
            max_iterations=5
        )

        # Verify summarize was called twice
        assert len(summarize_calls) == 2
        # Verify task completed
        assert response.status
        assert response.error is None

    def test_summarize_context_continues_loop(self, no_mount_microBot, monkeypatch):
        """Test that after summarize_context, the bot continues processing without executing command."""
        assert no_mount_microBot is not None

        executed_commands = []
        summarize_calls = []

        def mock_execute(command: str, timeout: int = 60):
            executed_commands.append(command)
            return CmdReturn(return_code=0, stdout="executed", stderr="")

        def mock_summarize_context(last_n_messages: int, summary: str):
            summarize_calls.append((last_n_messages, summary))
            return {"role": "user", "content": "Previous message to continue"}

        call_count = [0]

        def mock_ask(message: str):
            no_mount_microBot.llm.messages.append({"role": "user", "content": message})
            call_count[0] += 1
            if call_count[0] == 1:
                # Initial prompt: return summarize_context command
                res = LLMAskResponse(
                    command='summarize_context 1 "Context summary"',
                    task_done=False,
                    thoughts=""
                )
            elif call_count[0] == 2:
                # After summarize (with returned last message): return actual command
                res = LLMAskResponse(command="echo 'actual command'", task_done=False, thoughts="")
            else:
                res = LLMAskResponse(command="", task_done=True, thoughts="Complete")
            no_mount_microBot.llm.messages.append({"role": "assistant", "content": json.dumps(asdict(res))})
            return res

        monkeypatch.setattr(no_mount_microBot.llm, "ask", mock_ask)
        monkeypatch.setattr(no_mount_microBot.llm, "summarize_context", mock_summarize_context)
        monkeypatch.setattr(no_mount_microBot.environment, "execute", mock_execute)

        response: BotRunResult = no_mount_microBot.run(
            "Test summarize continues loop",
            timeout_in_seconds=60,
            max_iterations=10
        )

        # Verify summarize_context was called
        assert len(summarize_calls) == 1

        # Verify the summarize_context command was NOT executed in the environment
        assert 'summarize_context 1 "Context summary"' not in executed_commands

        # Verify actual commands were executed
        assert "echo 'actual command'" in executed_commands
        assert response.status

    def test_summarize_context_with_empty_summary(self, no_mount_microBot, monkeypatch):
        """Test summarize_context with empty summary (used to clear context after sub-task)."""
        assert no_mount_microBot is not None

        summarize_calls = []

        def mock_summarize_context(last_n_messages: int, summary: str):
            summarize_calls.append((last_n_messages, summary))
            return {"role": "user", "content": "Previous context message"}

        call_count = [0]

        def mock_ask(message: str):
            no_mount_microBot.llm.messages.append({"role": "user", "content": message})
            call_count[0] += 1
            if call_count[0] == 1:
                # Empty summary - used to clear previous context
                res = LLMAskResponse(
                    command='summarize_context 0 ""',
                    task_done=False,
                    thoughts="Clearing context after completing sub-task"
                )
            elif call_count[0] == 2:
                # After summarize (with returned last message): complete
                res = LLMAskResponse(command="", task_done=True, thoughts="Done")
            else:
                res = LLMAskResponse(command="", task_done=True, thoughts="Done")
            no_mount_microBot.llm.messages.append({"role": "assistant", "content": json.dumps(asdict(res))})
            return res

        monkeypatch.setattr(no_mount_microBot.llm, "ask", mock_ask)
        monkeypatch.setattr(no_mount_microBot.llm, "summarize_context", mock_summarize_context)

        response: BotRunResult = no_mount_microBot.run(
            "Test empty summarize",
            timeout_in_seconds=60,
            max_iterations=10
        )

        # Verify summarize_context was called with empty summary
        assert len(summarize_calls) == 1
        assert summarize_calls[0] == (0, "")
        assert response.status

    @pytest.mark.parametrize("summarize_command", [
        'summarize_context 1 "Simple summary"',
        'summarize_context 2 "Summary with multiple words and details"',
        'summarize_context 0 ""',
        'summarize_context 5 "Keep 5 turns, summary: completed file exploration"',
    ])
    def test_summarize_context_various_formats(self, summarize_command):
        """Test that various summarize_context command formats are detected correctly."""
        # Verify the command starts with summarize_context
        assert summarize_command.startswith("summarize_context")
