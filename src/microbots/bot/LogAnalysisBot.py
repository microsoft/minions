from typing import Optional

from microbots.constants import DOCKER_WORKING_DIR, LOG_FILE_DIR, PermissionLabels
from microbots.MicroBot import (
    BotType,
    MicroBot,
    get_file_mount_info,
    get_folder_mount_info,
    system_prompt_common,
)
from microbots.tools.tool import Tool


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
        permission = PermissionLabels.READ_ONLY

        folder_mount_info = get_folder_mount_info(folder_to_mount)
        base_name = folder_mount_info.base_name

        system_prompt = f"""
        {system_prompt_common}
        You are a log analysis bot.
        You are only provided access to read files inside the mounted directory.
        The directory is mounted at /{DOCKER_WORKING_DIR}/{base_name} in your current environment.
        You can access files using paths like /{DOCKER_WORKING_DIR}/{base_name}/filename.txt or by changing to that directory first.
        As part of the task you will be given a 
        Once all the commands are done, and task is verified finally give me the result.
        Also in the upcoming prompts you will be given a specific log file to analyze in the directory {LOG_FILE_DIR}
        """

        super().__init__(
            bot_type,
            model,
            system_prompt,
            environment,
            additional_tools,
            folder_to_mount,
            permission,
        )

    def __run__(self, file_name: str):

        # Add the logic to copy the file from the user path to /var/log path in container

        file_mount_info = get_file_mount_info(file_name)
        if not file_mount_info.path_valid:
            raise ValueError(f"file name {file_name} is not a valid path")

        # Copy the file to the container
        self.environment.copy_to_container(
            file_mount_info.abs_path, f"/var/log/{file_mount_info.base_name}"
        )

        file_name_prompt = f"""
        The log file to analyze is {LOG_FILE_DIR}/{file_name}
        """
        return self.run(file_name_prompt, file_name)
