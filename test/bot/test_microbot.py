"""
This test uses the Microbot base class to create a custom bot and tries to solve
https://github.com/SWE-agent/test-repo/issues/1.
This test will create multiple custom bots - a reading bot, a writing bot using the base class.
"""

import os
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
from microbots.MicroBot import BotRunResult
from microbots.constants import DOCKER_WORKING_DIR, PermissionLabels
from microbots.utils.env_mount import Mount, MountType # TODO Mount and MountType should be exposed from generic place


SYSTEM_PROMPT = f"""
You are a helpful python programmer who is good in debugging code.
You have the python repo where you're working mounted at {DOCKER_WORKING_DIR}.
You have a shell session open for you.
I will provide a task to achieve using the shell commands.
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
            subprocess.run(["rm", "-rf", str(tmpdir)])

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
            f"The test file tests/missing_colon.py is failing. Please fix the code. The error log is available at /var/log/{log_file_path.name}.",
            additional_mounts=[additional_mounts],
            timeout_in_seconds=300
        )

        print(f"Custom Coding Bot - Status: {response.status}, Result: {response.result}, Error: {response.error}")

        assert response.status
        assert response.result is not None
        assert response.error is None

        verify_function(test_repo)
