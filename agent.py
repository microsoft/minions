# agent.py

# agent_type can be either "READING_AGENT" | "WRITING_AGENT" | "BROWSING_AGENT" | "CUSTOM_AGENT"
# Create enum for agent_type and assign to agent argument agent_type
import time
from typing import Optional

from agent_utils import (
    AgentRunResult,
    AgentType,
    Model,
    ModelProvider,
    PermissionLabels,
    PermissionMapping,
    get_system_prompt,
    llm_finished_keyword,
    validate_agent_type_and_folder_to_mount,
    validate_agent_type_and_permission,
    validate_agent_type_and_system_prompt,
)
from llm.openai_api import OpenAIResponseApi

tools = []


class Agent:

    def __init__(
        self,
        agent_type: AgentType,
        model_details: Model,
        system_prompt: Optional[str] = None,
        environment: Optional[any] = None,  # TODO: Need to pass environment class
        additional_tools: Optional[str] = None,
        folder_to_mount: Optional[str] = None,
        permission: Optional[PermissionLabels] = None,
    ):
        # validate init values before assigning
        validate_agent_type_and_system_prompt(agent_type, system_prompt)
        validate_agent_type_and_permission(agent_type, permission)
        validate_agent_type_and_folder_to_mount(agent_type, folder_to_mount)

        # assign values to class variables
        self.agent_type = agent_type
        if permission is None and agent_type == AgentType.CUSTOM_AGENT:
            self.permission = PermissionLabels.READ_WRITE
        else:
            self.permission = permission

        self.permission_key = PermissionMapping.MAPPING.get(self.permission)
        self.system_prompt = get_system_prompt(
            agent_type, folder_to_mount, system_prompt
        )
        self.environment = environment  # TODO: initialize environment
        self.tools = tools + additional_tools
        self.model_details = model_details

        # initialize llm with system prompt
        if self.model_details.provider == ModelProvider.OPENAI:

            self.llm = OpenAIResponseApi(
                system_prompt=self.system_prompt,
                deployment_name=self.model_details.deployment_name,
            )

        # initialize the environment and install all the tools inside the environment
        # and add the tool usage instructions to system prompt
        # also convey about the mounted directory and permission to the llm on the mounted directory

    # return type of run should be { status: bool, result: string, error: string | None, log: log of communication between llm and environment }
    def run(self, task, max_iterations=20, timeout_in_seconds=10) -> AgentRunResult:
        iteration_count = 0
        # start timer
        start_time = time.time()
        timeout = timeout_in_seconds
        llm_response = self.llm.ask(task)
        return_value: AgentRunResult = {
            "status": False,
            "error": "Did not complete",
        }

        while llm_response.task_done is False:

            llm_command_output = self.environment.execute(llm_response.command)
            llm_response = self.llm.ask(llm_command_output)

            # increment iteration count
            iteration_count += 1
            if iteration_count > max_iterations:
                return_value.error = f"Max iterations {max_iterations} reached"
                return return_value

            # check if timeout has reached
            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time > timeout:
                return_value.error = f"Timeout of {timeout} seconds reached"
                return return_value

        return {"status": True, "result": llm_response.result}
