import json
import os
import time
from dataclasses import dataclass
from enum import StrEnum
from logging import getLogger
from typing import Optional

from constants import ModelProvider, PermissionLabels, PermissionMapping
from environment.local_docker.LocalDockerEnvironment import LocalDockerEnvironment
from llm.openai_api import OpenAIApi
from tool_definitions.base_tool import BaseTool
from utils.logger import LogLevelEmoji, dividerString
from utils.network import get_free_port

logger = getLogger(" Minion ")

llm_output_format = """```json
{
    task_done: true | false,
    command: "<command to run> | null",
    result: str | null
}
```
"""

system_prompt_common = """There is a shell session open for you. 
                I will provide a task to achieve using the shell. 
                You will provide the commands to achieve the task in this particular below json format, Ensure all the time to respond in this format only and nothing else, also all the properties ( task_done, command, result ) are mandatory on each response
                {llm_output_format}
                after each command I will provide the output of the command. 
                ensure to run only one command at a time.
                I won't be able to intervene once I have given task. ."""


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
        self.permission = permission
        if folder_to_mount is not None:
            self.folder_to_mount_base_path = os.path.basename(folder_to_mount)  # TODO

        self._validate_model_and_provider(model)
        self.permission_key = PermissionMapping.MAPPING.get(self.permission)
        self.system_prompt = system_prompt
        self.model = model
        self.agent_type = agent_type
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

    def _validate_model_and_provider(self, model):
        # Ensure it has only only slash
        if model.count("/") != 1:
            raise ValueError("Model should be in the format <provider>/<model_name>")
        provider = model.split("/")[0]
        if provider not in [e.value for e in ModelProvider]:
            raise ValueError(f"Unsupported model provider: {provider}")
