import os
from typing import Optional

from agent import AgentType, Minion, system_prompt_common
from constants import PermissionLabels
from tool_definitions.base_tool import BaseTool


class BrowserAgent(Minion):

    def __init__(
        self,
        model: str,
        environment: Optional[any] = None,
        additional_tools: Optional[list[BaseTool]] = [],
    ):
        # validate init values before assigning
        agent_type = AgentType.BROWSING_AGENT
        permission = PermissionLabels.READ_WRITE
        system_prompt = f"""
        {system_prompt_common}
        You are also provided access to internet to search for information.
        """

        super().__init__(
            agent_type,
            model,
            system_prompt,
            environment,
            additional_tools,
            permission,
        )
