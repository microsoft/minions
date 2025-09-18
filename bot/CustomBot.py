from typing import Optional

from constants import PermissionLabels
from MicroBot import BotType, MicroBot
from tools.tool import Tool


class BrowserBot(MicroBot):

    def __init__(
        self,
        model: str,
        system_prompt: str,
        folder_to_mount: Optional[str] = None,
        environment: Optional[any] = None,
        additional_tools: Optional[list[Tool]] = [],
    ):
        # validate init values before assigning
        bot_type = BotType.BROWSING_BOT
        permission = PermissionLabels.READ_WRITE

        super().__init__(
            bot_type,
            model,
            system_prompt,
            environment,
            additional_tools,
            folder_to_mount,
            permission,
        )
