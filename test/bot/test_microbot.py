"""
This test uses the Microbot base class to create a custom bot and tries to solve
https://github.com/SWE-agent/test-repo/issues/1.
This test will create multiple custom bots - a reading bot, a writing bot using the base class.
"""

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
    bot = MicroBot(
        model="ollama-local/qwen2.5-coder",
        system_prompt=SYSTEM_PROMPT,
    )
    yield bot
    del bot


@pytest.mark.integration
@pytest.mark.docker
@pytest.mark.slow
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
        bot = MicroBot(
            model="ollama-local/qwen2.5-coder",
            system_prompt=SYSTEM_PROMPT,
            folder_to_mount=ro_mount,
        )
        yield bot
        del bot

    @pytest.fixture(scope="function")
    def anthropic_microBot(self):
        with patch('microbots.llm.anthropic_api.endpoint', 'https://api.anthropic.com'), \
             patch('microbots.llm.anthropic_api.deployment_name', 'claude-sonnet-4'), \
             patch('microbots.llm.anthropic_api.api_key', 'test-api-key'), \
             patch('microbots.llm.anthropic_api.Anthropic'):
            bot = MicroBot(
                model="anthropic/claude-sonnet-4",
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
        assert anthropic_microBot.deployment_name == "claude-sonnet-4"
        assert anthropic_microBot.llm is not None
        from microbots.llm.anthropic_api import AnthropicApi
        assert isinstance(anthropic_microBot.llm, AnthropicApi)

    def test_microbot_2bot_combo(self, log_file_path, test_repo, issue_1):
        assert test_repo is not None
        assert log_file_path is not None

        verify_function = issue_1[1]

        test_repo_mount_ro = Mount(
            str(test_repo),
            f"{DOCKER_WORKING_DIR}/{test_repo.name}",
            PermissionLabels.READ_ONLY
        )
        testing_bot = MicroBot(
            model="azure-openai/mini-swe-agent-gpt5",
            system_prompt=SYSTEM_PROMPT,
            folder_to_mount=test_repo_mount_ro,
        )

        response: BotRunResult = testing_bot.run(
            "Execute tests/missing_colon.py and provide the error message. Your response should be in 'thoughts' field.",
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
        coding_bot = MicroBot(
            model="azure-openai/mini-swe-agent-gpt5",
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
            f"The test file tests/missing_colon.py is failing. Please fix the code. The error log is available at /var/log/{log_file_path.basename}.",
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
        with patch('microbots.llm.anthropic_api.endpoint', 'https://api.anthropic.com'), \
             patch('microbots.llm.anthropic_api.deployment_name', 'claude-sonnet-4'), \
             patch('microbots.llm.anthropic_api.api_key', 'test-api-key'), \
             patch('microbots.llm.anthropic_api.Anthropic'):
            bot = MicroBot(
                model="anthropic/claude-sonnet-4",
                system_prompt=SYSTEM_PROMPT,
                folder_to_mount=test_repo_mount_ro,
            )
            assert bot is not None
            assert bot.model_provider == "anthropic"
            from microbots.llm.anthropic_api import AnthropicApi
            assert isinstance(bot.llm, AnthropicApi)
            del bot


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

        with pytest.raises(ValueError, match="Only MOUNT mount type is supported for folder_to_mount"):
            MicroBot(
                model="ollama-local/qwen2.5-coder",
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
            return LLMAskResponse(command="echo 'Hello World'", task_done=False, thoughts="")

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
            return LLMAskResponse(command="sleep 10", task_done=False, thoughts="")

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
            call_count[0] += 1
            if call_count[0] == 1:
                # First call returns dangerous command
                return LLMAskResponse(command="ls -R /path", task_done=False, thoughts="")
            else:
                # After receiving error with explanation, return safe command
                assert "COMMAND_ERROR:" in message
                assert "Dangerous command detected and blocked" in message
                assert "REASON:" in message
                assert "ALTERNATIVE:" in message
                return LLMAskResponse(command="pwd", task_done=True, thoughts="")

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
