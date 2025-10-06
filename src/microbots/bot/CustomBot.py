from typing import Any, Optional

from microbots.constants import DOCKER_WORKING_DIR, PermissionLabels
from microbots.MicroBot import BotType, MicroBot
from microbots.tools.tool import Tool
from microbots.utils.env_mount import Mount


class CustomBot(MicroBot):
    """
    Customizable bot with user-defined system prompt and bot type.

    CustomBot allows full customization of behavior through custom system prompts
    while maintaining the standard bot execution flow. Provides read-write access
    to mounted directories.

    Args:
        model: Model string in format "<provider>/<deployment_name>"
        system_prompt: Custom system prompt defining the bot's behavior
        folder_to_mount: Optional absolute path to directory to mount with read-write access
        environment: Optional custom environment instance
        additional_tools: Optional list of additional Tool instances
    """

    def __init__(
        self,
        model: str,
        system_prompt: str,
        folder_to_mount: Optional[str] = None,
        environment: Optional[Any] = None,
        additional_tools: Optional[list[Tool]] = None,
    ):
        if additional_tools is None:
            additional_tools = []

        bot_type = BotType.CUSTOM_BOT

        # Create Mount object if folder_to_mount is provided
        folder_mount_info = None
        if folder_to_mount is not None:
            folder_mount_info = Mount(
                folder_to_mount, f"/{DOCKER_WORKING_DIR}", PermissionLabels.READ_WRITE
            )

        super().__init__(
            bot_type,
            model,
            system_prompt,
            environment,
            additional_tools,
            folder_mount_info,
        )
