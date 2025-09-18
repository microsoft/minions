from typing import Optional
from pathlib import Path

from agent import AgentType, Minion, system_prompt_common
from constants import PermissionLabels
from tool_definitions.base_tool import BaseTool


class ReadingAgent(Minion):

    def __init__(
        self,
        model: str,
        folder_to_mount: str,
        environment: Optional[any] = None,
        additional_tools: Optional[list[BaseTool]] = [],
    ):
        # validate init values before assigning
        agent_type = AgentType.READING_AGENT
        permission = PermissionLabels.READ_ONLY

        base_name = Path(folder_to_mount).name

        system_prompt = f"""
        {system_prompt_common}
        You are a reading agent.
        You are only provided access to read files inside the mounted directory.
        The directory is mounted at /app/{base_name} in your current environment.
        You can access files using paths like /app/{base_name}/filename.txt or by changing to that directory first.
        Once all the commands are done, and task is verified finally give me the result.
        """

        super().__init__(
            agent_type,
            model,
            system_prompt,
            environment,
            additional_tools,
            folder_to_mount,
            permission,
        )
