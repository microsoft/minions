# agent.py

# agent_type can be either "READING_AGENT" | "WRITING_AGENT" | "BROWSING_AGENT" | "CUSTOM_AGENT"
# Create enum for agent_type and assign to agent argument agent_type
import logging
import os
import time
from typing import Optional
from agent_utils import (
    AgentRunResult,
    AgentType,
    ModelProvider,
    PermissionLabels,
    PermissionMapping,
    assign_permission_based_on_agent_type,
    get_default_tools,
    get_free_port,
    get_system_prompt,
    validate_agent_type_and_folder_to_mount,
    validate_agent_type_and_permission,
    validate_agent_type_and_system_prompt,
)
from Environment.LocalDockerEnvironment import LocalDockerEnvironment
from llm.openai_api import OpenAIApi
from tool_definitions.base_tool import BaseTool

tools = []
logger = logging.getLogger(__name__)


class Agent:

    def __init__(
        self,
        agent_type: AgentType,
        model: str,
        system_prompt: Optional[str] = None,
        environment: Optional[any] = None,  # TODO: Need to pass environment class
        additional_tools: Optional[list[BaseTool]] = [],
        folder_to_mount: Optional[str] = None,
        permission: Optional[PermissionLabels] = None,
    ):
        # validate init values before assigning
        self.permission = assign_permission_based_on_agent_type(agent_type, permission)
        validate_agent_type_and_system_prompt(agent_type, system_prompt)
        validate_agent_type_and_permission(agent_type, permission)
        validate_agent_type_and_folder_to_mount(agent_type, folder_to_mount)

        if folder_to_mount is not None:
            self.folder_to_mount_base_path = os.path.basename(folder_to_mount)
        self.permission_key = PermissionMapping.MAPPING.get(self.permission)
        self.system_prompt = get_system_prompt(
            agent_type, self.folder_to_mount_base_path, system_prompt
        )
        self.tools = get_default_tools() + additional_tools
        self.model = model

        # model will be a string like "openai/gpt-5"
        self.model_provider = model.split("/")[0]
        self.deployment_name = model.split("/")[1]
        self.environment = environment
        self.tool_usage_instructions = ""

        # initialize the environment and install all the tools inside the environment
        if self.environment is None:
            # check for a free port in the system and assign to environment

            free_port = get_free_port()

            self.environment = LocalDockerEnvironment(
                port=free_port,
                folder_to_mount=folder_to_mount,
                permission=self.permission,
            )

        # initialize llm with system prompt
        if self.model_provider == ModelProvider.OPENAI:

            self.llm = OpenAIApi(
                system_prompt=self.system_prompt, deployment_name=self.deployment_name
            )

        # for tool in self.tools:
        # logger.info(f"⬇️ Installing tool: {tool.name}")
        # installation_status = self._tool_installation_through_ai(tool)
        # if installation_status is False:
        #     raise Exception(f"Tool {tool.name} installation failed")
        # logger.info(f"✅ Tool {tool.name} installed successfully")
        # self.tool_usage_instructions += tool.usage_instructions_to_llm + "\n"

        # self.llm.clear_history()

        self.system_prompt += (
            "\nThere are some special tool commands available in shell. I have mentioned those with it's usage instructions below \n. Ensure "
            + self.tool_usage_instructions
            + "\n"
        )

    def run(self, task, max_iterations=20, timeout_in_seconds=200) -> AgentRunResult:

        iteration_count = 0
        # start timer
        start_time = time.time()
        timeout = timeout_in_seconds
        llm_response = self.llm.ask(task)
        return_value: AgentRunResult = AgentRunResult(
            status=False,
            result=None,
            error="Did not complete",
        )

        while llm_response["task_done"] is False:

            # increment iteration count
            iteration_count += 1
            if iteration_count >= max_iterations:
                return_value["error"] = f"Max iterations {max_iterations} reached"
                return return_value

            # check if timeout has reached
            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time > timeout:
                return_value["error"] = f"Timeout of {timeout} seconds reached"
                return return_value

            llm_command_output = self.environment.execute(llm_response["command"])
            llm_response = self.llm.ask(llm_command_output)

        return AgentRunResult(
            status=True,
            result=llm_response["result"],
            error=None,
        )

    def _tool_installation_through_ai(self, tool: BaseTool):
        result = self.run(
            f"""The task is to install the tool using the command: {tool.installation_command}
                 you can verify if the tool is installed correctly using the following command: {tool.verification_command}
                 """
        )
        return result["status"]
