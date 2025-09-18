import json
import os
import time
from dataclasses import dataclass
from enum import StrEnum
from logging import getLogger
from typing import Optional

from utils.logger import LogLevelEmoji, dividerString
from utils.network import get_free_port

from constants import ModelProvider, PermissionLabels, PermissionMapping
from environment.local_docker.LocalDockerEnvironment import LocalDockerEnvironment
from llm.openai_api import OpenAIApi
from tool_definitions.base_tool import BaseTool

logger = getLogger(" Minion ")

llm_output_format = """```json
{
    task_done: true | false,
    command: "<command to run> | null",
    result: str | null
}
```
"""


class AgentType(StrEnum):
    READING_AGENT = "READING_AGENT"
    WRITING_AGENT = "WRITING_AGENT"
    BROWSING_AGENT = "BROWSING_AGENT"
    CUSTOM_AGENT = "CUSTOM_AGENT"


@dataclass
class AgentRunResult:
    status: bool
    result: str | None
    error: Optional[str]


class Minion:

    def __init__(
        self,
        agent_type: AgentType,
        model: str,
        system_prompt: Optional[str] = None,
        environment: Optional[any] = None,
        additional_tools: Optional[list[BaseTool]] = [],
        folder_to_mount: Optional[str] = None,
        permission: Optional[PermissionLabels] = None,
    ):
        # validate init values before assigning
        self.permission = self._assign_permission_based_on_agent_type(
            agent_type, permission
        )
        self._agent_arg_validation(
            agent_type, system_prompt, self.permission, folder_to_mount, model
        )
        if folder_to_mount is not None:
            self.folder_to_mount_base_path = os.path.basename(folder_to_mount)  # TODO

        self.permission_key = PermissionMapping.MAPPING.get(self.permission)
        self.system_prompt = self._get_system_prompt(
            agent_type, self.folder_to_mount_base_path, system_prompt
        )
        self.model = model
        self.model_provider = model.split("/")[0]
        self.deployment_name = model.split("/")[1]
        self.environment = environment
        self._create_environment(folder_to_mount)
        self._create_llm()

    def run(self, task, max_iterations=20, timeout_in_seconds=200) -> AgentRunResult:

        iteration_count = 1
        # start timer
        start_time = time.time()
        timeout = timeout_in_seconds
        llm_response = self.llm.ask(task)
        return_value = AgentRunResult(
            status=False,
            result=None,
            error="Did not complete",
        )
        logger.info("%s TASK STARTED : %s...", LogLevelEmoji.INFO, task[0:15])
        while llm_response.task_done is False:
            print(dividerString)
            logger.info(
                " %s LLM Iteration Count : %d", LogLevelEmoji.INFO, iteration_count
            )
            logger.info(
                " %s LLM tool call : %s",
                LogLevelEmoji.INFO,
                json.dumps(llm_response.command),
            )
            # increment iteration count
            iteration_count += 1
            if iteration_count >= max_iterations:
                return_value.error = f"Max iterations {max_iterations} reached"
                return return_value

            # check if timeout has reached
            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time > timeout:
                logger.error(
                    "Iteration %d with response %s",
                    iteration_count,
                    json.dumps(llm_response),
                )
                return_value.error = f"Timeout of {timeout} seconds reached"
                return return_value

            llm_command_output = self.environment.execute(llm_response.command)
            logger.info(
                " %s Command Execution Output : %s",
                LogLevelEmoji.INFO,
                llm_command_output,
            )
            llm_response = self.llm.ask(llm_command_output)

        logger.info("%s TASK COMPLETED : %s...", LogLevelEmoji.COMPLETED, task[0:15])
        return AgentRunResult(status=True, result=llm_response.result, error=None)

    def _create_environment(self, folder_to_mount):
        if self.environment is None:
            # check for a free port in the system and assign to environment

            free_port = get_free_port()

            self.environment = LocalDockerEnvironment(
                port=free_port,
                folder_to_mount=folder_to_mount,
                permission=self.permission,
            )

    def _create_llm(self):
        if self.model_provider == ModelProvider.OPENAI:
            self.llm = OpenAIApi(
                system_prompt=self.system_prompt, deployment_name=self.deployment_name
            )

    def _agent_arg_validation(
        self, agent_type, system_prompt, permission, folder_to_mount, model
    ):
        self._validate_agent_type_and_system_prompt(agent_type, system_prompt)
        self._validate_agent_type_and_permission(agent_type, permission)
        self._validate_agent_type_and_folder_to_mount(agent_type, folder_to_mount)
        self._validate_model_and_provider(model)

    def _get_system_prompt(
        self, agent_type: AgentType, mounted_directory: str, system_prompt: str | None
    ) -> str:

        # Create system prompt based on agent_type and permission mapping
        system_prompt_common = """There is a shell session open for you.
                I will provide a task to achieve using the shell.
                You will provide the commands to achieve the task in this particular below json format, Ensure all the time to respond in this format only and nothing else, also all the properties ( task_done, command, result ) are mandatory on each response
                {llm_output_format}
                after each command I will provide the output of the command.
                ensure to run only one command at a time.
                I won't be able to intervene once I have given task. ."""

        system_prompts = {
            AgentType.READING_AGENT: f"""
        {system_prompt_common}
        You are a reading agent.
        You are only provided access only read files inside the mounted directory {mounted_directory}.
        Once all the commands are done, and task is verified finally give me .
        """,
            AgentType.WRITING_AGENT: f"""
        {system_prompt_common}
        You are a writing agent.
        You are provided access to read and write files inside the mounted directory {mounted_directory}.
        """,
            AgentType.BROWSING_AGENT: f"""
        {system_prompt_common}
        You are also provided access to internet to search for information.
        """,
            AgentType.CUSTOM_AGENT: system_prompt,
        }

        return system_prompts.get(agent_type)

    def _assign_permission_based_on_agent_type(self, agent_type, permission):
        return_value = permission
        if permission is None and agent_type == AgentType.CUSTOM_AGENT:
            return_value = PermissionLabels.READ_WRITE
        else:
            if agent_type == AgentType.READING_AGENT:
                return_value = PermissionLabels.READ_ONLY
            elif agent_type == AgentType.WRITING_AGENT:
                return_value = PermissionLabels.READ_WRITE
        return return_value

    def _validate_agent_type_and_system_prompt(self, agent_type, system_prompt):
        if agent_type == AgentType.CUSTOM_AGENT and system_prompt is None:
            raise ValueError("Custom agent requires a system prompt")

        elif (
            agent_type == AgentType.READING_AGENT
            or agent_type == AgentType.BROWSING_AGENT
            or agent_type == AgentType.WRITING_AGENT
        ) and system_prompt is not None:
            raise ValueError(
                "System prompt should not be provided for non-custom agents"
            )

    def _validate_agent_type_and_permission(self, agent_type, permission):
        if (
            agent_type == AgentType.READING_AGENT
            and permission == PermissionLabels.READ_WRITE
        ):
            raise ValueError("Reading agent cannot have read-write permission")

        elif (
            agent_type == AgentType.WRITING_AGENT
            and permission == PermissionLabels.READ_ONLY
        ):
            raise ValueError("Writing agent cannot have read-only permission")

        elif agent_type == AgentType.BROWSING_AGENT and permission is not None:
            raise ValueError("Browsing agent cannot have permission provided")

    def _validate_agent_type_and_folder_to_mount(self, agent_type, folder_to_mount):

        if agent_type == AgentType.CUSTOM_AGENT and folder_to_mount is None:
            raise ValueError("Folder to mount is required for custom agent")

        elif agent_type == AgentType.WRITING_AGENT and folder_to_mount is None:
            raise ValueError("Folder to mount should provided for writing agents")

        elif agent_type == AgentType.READING_AGENT and folder_to_mount is None:
            raise ValueError("Folder to mount should provided for reading agents")

        elif agent_type == AgentType.BROWSING_AGENT and folder_to_mount is not None:
            raise ValueError(
                "Folder to mount should not be provided for browsing agents"
            )

    def _validate_model_and_provider(self, model):
        # Ensure it has only only slash
        if model.count("/") != 1:
            raise ValueError("Model should be in the format <provider>/<model_name>")
        provider = model.split("/")[0]
        if provider not in [e.value for e in ModelProvider]:
            raise ValueError(f"Unsupported model provider: {provider}")
