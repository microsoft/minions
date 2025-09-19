import os
from typing import Optional

from microbots.constants import PermissionLabels
from microbots.MicroBot import BotType, MicroBot, system_prompt_common
from microbots.tool_definitions.base_tool import BaseTool


class BrowserBot(MicroBot):

    def __init__(
        self,
        model: str,
        environment: Optional[any] = None,
        additional_tools: Optional[list[BaseTool]] = [],
    ):
        # validate init values before assigning
        bot_type = BotType.BROWSING_BOT
        permission = PermissionLabels.READ_WRITE
        system_prompt = f"""
        {system_prompt_common}
        You are also provided access to internet to search for information.
        """

        super().__init__(
            bot_type,
            model,
            system_prompt,
            environment,
            additional_tools,
            permission,
        )
