from typing import Optional

from microbots.constants import DOCKER_WORKING_DIR, PermissionLabels
from microbots.MicroBot import BotType, MicroBot, get_path_info, system_prompt_common
from microbots.tools.tool import Tool


class ReadingBot(MicroBot):

    def __init__(
        self,
        model: str,
        folder_to_mount: str,
        environment: Optional[any] = None,
        additional_tools: Optional[list[Tool]] = [],
    ):
        # validate init values before assigning
        bot_type = BotType.READING_BOT
        permission = PermissionLabels.READ_ONLY

        folder_mount_info = get_path_info(folder_to_mount)
        base_name = folder_mount_info.base_name

        system_prompt = f"""
        {system_prompt_common}
        You are a reading bot.
        You are only provided access to read files inside the mounted directory.
        The directory is mounted at /{DOCKER_WORKING_DIR}/{base_name} in your current environment.
        You can access files using paths like /{DOCKER_WORKING_DIR}/{base_name}/filename.txt or by changing to that directory first.
        Once all the commands are done, and task is verified finally give me the result.
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
