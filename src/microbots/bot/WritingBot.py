from typing import Optional

from microbots.constants import DOCKER_WORKING_DIR, PermissionLabels
from microbots.MicroBot import BotType, MicroBot, system_prompt_common
from microbots.tools.tool import Tool
from microbots.utils.env_mount import Mount


class WritingBot(MicroBot):

    def __init__(
        self,
        model: str,
        folder_to_mount: str,
        environment: Optional[any] = None,
        additional_tools: Optional[list[Tool]] = [],
    ):
        # validate init values before assigning
        bot_type = BotType.WRITING_BOT

        folder_mount_info = Mount(
            folder_to_mount, DOCKER_WORKING_DIR, PermissionLabels.READ_WRITE
        )

        system_prompt = f"""
        {system_prompt_common}
        You are a writing bot.
        You are only provided access to write files inside the mounted directory.
        The directory is mounted at  {folder_mount_info.sandbox_path} in your current environment.
        You can access files using paths like {folder_mount_info.sandbox_path}/filename.txt or by changing to that directory first.
        Once all the commands are done, and task is verified finally give me the result.
        """

        super().__init__(
            bot_type,
            model,
            system_prompt,
            environment,
            additional_tools,
            folder_mount_info,
        )
