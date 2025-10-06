from typing import Any, Optional

from microbots.constants import DOCKER_WORKING_DIR, PermissionLabels
from microbots.MicroBot import BotType, MicroBot, SYSTEM_PROMPT_BASE
from microbots.tools.tool import Tool
from microbots.utils.env_mount import Mount


class ReadingBot(MicroBot):
    """
    Bot with read-only access to a mounted directory.

    ReadingBot creates a Docker environment with read-only file access using
    an overlay filesystem. The LLM can read files but any write attempts
    are isolated and don't affect the original directory.

    Args:
        model: Model string in format "<provider>/<deployment_name>"
        folder_to_mount: Absolute path to the directory to mount as read-only
        environment: Optional custom environment instance
        additional_tools: Optional list of additional Tool instances
    """

    def __init__(
        self,
        model: str,
        folder_to_mount: str,
        environment: Optional[Any] = None,
        additional_tools: Optional[list[Tool]] = None,
    ):
        # validate init values before assigning
        bot_type = BotType.READING_BOT

        if additional_tools is None:
            additional_tools = []

        folder_mount_info = Mount(
            folder_to_mount, f"/{DOCKER_WORKING_DIR}", PermissionLabels.READ_ONLY
        )

        system_prompt = f"""{SYSTEM_PROMPT_BASE}

ROLE: You are a reading bot with read-only access to files.

MOUNTED DIRECTORY:
- Location: {folder_mount_info.sandbox_path}
- Access: Read-only (you cannot modify files)
- Usage: Access files using {folder_mount_info.sandbox_path}/filename.txt

Once all commands are executed and the task is verified, provide the final result."""

        super().__init__(
            bot_type,
            model,
            system_prompt,
            environment,
            additional_tools,
            folder_mount_info,
        )
