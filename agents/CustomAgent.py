import os
from typing import Optional

from agent import AgentType, Minion
from constants import PermissionLabels
from tool_definitions.base_tool import BaseTool


class BrowserAgent(Minion):

    def __init__(
        self,
        model: str,
        system_prompt: str,
        folder_to_mount: Optional[str] = None,
        environment: Optional[any] = None,
        additional_tools: Optional[list[BaseTool]] = [],
    ):
        # validate init values before assigning
        agent_type = AgentType.BROWSING_AGENT
        permission = PermissionLabels.READ_WRITE

        super().__init__(
            agent_type,
            model,
            system_prompt,
            environment,
            additional_tools,
            folder_to_mount,
            permission,
        )
