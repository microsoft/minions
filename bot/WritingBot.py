from typing import Optional

from constants import PermissionLabels
from MicroBot import BotType, MicroBot, system_prompt_common
from tool_definitions.base_tool import BaseTool


class WritingBot(MicroBot):

    def __init__(
        self,
        model: str,
        folder_to_mount: str,
        environment: Optional[any] = None,
        additional_tools: Optional[list[BaseTool]] = [],
    ):
        # validate init values before assigning
        bot_type = BotType.WRITING_BOT
        permission = PermissionLabels.READ_WRITE

        system_prompt = f"""
        {system_prompt_common}
        You are a writing bot. 
        You are only provided access to write files inside the mounted directory.
        The directory is mounted at /app/{folder_to_mount} in your current environment.
        You can access files using paths like /app/{folder_to_mount}/filename.txt or by changing to that directory first.
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
