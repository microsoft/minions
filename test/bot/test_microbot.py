"""
This test uses the Microbot base class to create a custom bot and tries to solve
https://github.com/SWE-agent/test-repo/issues/1.
This test will create multiple custom bots - a reading bot, a writing bot using the base class.
"""

import os
from pathlib import Path
import subprocess
import sys

import pytest
# Add src directory to path to import from local source
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from microbots import MicroBot
from microbots.MicroBot import BotRunResult, llm_output_format
from microbots.constants import DOCKER_WORKING_DIR, PermissionLabels
from microbots.extras.mount import Mount, MountType
from microbots.environment.Environment import CmdReturn


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
{llm_output_format}
"""


@pytest.mark.integration
@pytest.mark.docker
@pytest.mark.slow
class TestMicroBot:

    @pytest.fixture(scope="function")
    def log_file_path(self, tmpdir):
        assert tmpdir.exists()
        yield tmpdir / "error.log"
        if tmpdir.exists():
            subprocess.run(["sudo", "rm", "-rf", str(tmpdir)])

    @pytest.fixture(scope="function")
    def ro_mount(self, test_repo):
        assert test_repo is not None
        return Mount(
            str(test_repo), DOCKER_WORKING_DIR, PermissionLabels.READ_ONLY
        )

    @pytest.fixture(scope="function")
    def ro_microBot(self, ro_mount: Mount):
        bot = MicroBot(
            model="azure-openai/mini-swe-agent-gpt5",
            system_prompt=SYSTEM_PROMPT,
            folder_to_mount=ro_mount,
        )
        yield bot
        del bot

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

    def test_microbot_overlay_teardown(self, ro_microBot, caplog):
        caplog.clear()
        caplog.set_level(logging.INFO)

        del ro_microBot

        assert "Failed to remove working directory" not in caplog.text

    def test_microbot_2bot_combo(self, log_file_path, test_repo, issue_1):
        assert test_repo is not None
        assert log_file_path is not None

        verify_function = issue_1[1]

        test_repo_mount_ro = Mount(
            str(test_repo), DOCKER_WORKING_DIR, PermissionLabels.READ_ONLY
        )
        testing_bot = MicroBot(
            model="azure-openai/mini-swe-agent-gpt5",
            system_prompt=SYSTEM_PROMPT,
            folder_to_mount=test_repo_mount_ro,
        )

        response: BotRunResult = testing_bot.run(
            "Execute tests/missing_colon.py and provide the error message",
            timeout_in_seconds=300
        )

        print(f"Custom Reading Bot - Status: {response.status}, Result: {response.result}, Error: {response.error}")

        assert response.status
        assert response.result is not None
        assert response.error is None

        with open(log_file_path, "w") as log_file:
            log_file.write(response.result)

        test_repo_mount_rw = Mount(
            str(test_repo), DOCKER_WORKING_DIR, PermissionLabels.READ_WRITE
        )
        coding_bot = MicroBot(
            model="azure-openai/mini-swe-agent-gpt5",
            system_prompt=SYSTEM_PROMPT,
            folder_to_mount=test_repo_mount_rw,
        )

        additional_mounts = Mount(
            log_file_path,
            "/var/log",
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

    def test_incorrect_mount_type(self, log_file_path, test_repo):
        assert test_repo is not None
        assert log_file_path is not None

        test_repo_mount_ro = Mount(
            str(test_repo), DOCKER_WORKING_DIR, PermissionLabels.READ_ONLY
        )
        testing_bot = MicroBot(
            model="azure-openai/mini-swe-agent-gpt5",
            system_prompt=SYSTEM_PROMPT,
            folder_to_mount=test_repo_mount_ro,
        )

        additional_mounts = Mount(
            log_file_path,
            "/var/log",
            PermissionLabels.READ_ONLY,
            MountType.MOUNT, # MOUNT is not supported yet
        )
        with pytest.raises(ValueError, match="Only COPY mount type is supported for additional mounts for now"):
            testing_bot.run(
                "Execute tests/missing_colon.py and provide the error message",
                additional_mounts=[additional_mounts],
                timeout_in_seconds=300
            )

    def test_incorrect_model_provider(self, test_repo):
        assert test_repo is not None

        test_repo_mount_ro = Mount(
            str(test_repo), DOCKER_WORKING_DIR, PermissionLabels.READ_ONLY
        )
        with pytest.raises(ValueError, match="Unsupported model provider: provider"):
            MicroBot(
                model="provider/invalidmodelname",
                system_prompt=SYSTEM_PROMPT,
                folder_to_mount=test_repo_mount_ro,
            )

    def test_incorrect_model_format(self, test_repo):
        assert test_repo is not None

        test_repo_mount_ro = Mount(
            str(test_repo), DOCKER_WORKING_DIR, PermissionLabels.READ_ONLY
        )
        with pytest.raises(ValueError, match="Model should be in the format <provider>/<model_name>"):
            MicroBot(
                model="invalidmodelname",
                system_prompt=SYSTEM_PROMPT,
                folder_to_mount=test_repo_mount_ro,
            )
