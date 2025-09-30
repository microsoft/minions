import logging
from typing import Optional

from microbots.constants import DOCKER_WORKING_DIR, LOG_FILE_DIR, PermissionLabels
from microbots.MicroBot import BotType, MicroBot, system_prompt_common
from microbots.tools.tool import Tool
from microbots.utils.env_mount import Mount, MountType
from microbots.utils.path import get_path_info

logger = logging.getLogger(__name__)


class LogAnalysisBot(MicroBot):

    def __init__(
        self,
        model: str,
        folder_to_mount: str,
        environment: Optional[any] = None,
        additional_tools: Optional[list[Tool]] = [],
    ):
        # validate init values before assigning
        bot_type = BotType.LOG_ANALYSIS_BOT

        folder_mount_info = Mount(
            folder_to_mount, DOCKER_WORKING_DIR, PermissionLabels.READ_ONLY
        )

        system_prompt = f"""
        {system_prompt_common}
        You are a helpful log analysis bot. Your job is to analyze a log file and identify the root-cause if there are any failure. You'll be given read-only access to the code from where the log is generated. The read-only code is available at {folder_mount_info.sandbox_path}.

The log file to analyze will be given in the user prompt. You can find the provided log file under the directory /{LOG_FILE_DIR}/

Only when you have run all necessary commands and identified the root cause, you should give me the final result.
        """

        super().__init__(
            bot_type,
            model,
            system_prompt,
            environment,
            additional_tools,
            folder_mount_info,
        )

    def run(self, file_name: str, timeout_in_seconds: int = 300) -> any:

        # Add the logic to copy the file from the user path to /var/log path in container
        file_mount_info = Mount(
            file_name,
            LOG_FILE_DIR,
            PermissionLabels.READ_ONLY,
            MountType.COPY,
        )
        self.mounted.append(file_mount_info)

        # Copy the file to the container
        copy_to_container_result = self.environment.copy_to_container(
            file_mount_info.host_path_info.abs_path, file_mount_info.sandbox_path
        )
        if copy_to_container_result is False:
            raise ValueError(
                f"Failed to copy file to container: {file_mount_info.host_path_info.base_name}"
            )

        file_name_prompt = f"""
            Analyze the log file `{file_mount_info.sandbox_path}`
        """
        return super().run(file_name_prompt, timeout_in_seconds)
