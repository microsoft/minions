import json
import os
import time
from dataclasses import dataclass
from enum import StrEnum
from logging import getLogger
from typing import Optional

from microbots.constants import ModelProvider, PermissionLabels, PermissionMapping
from microbots.environment.local_docker.LocalDockerEnvironment import (
    LocalDockerEnvironment,
)
from microbots.llm.openai_api import OpenAIApi
from microbots.tools.tool import Tool, install_tools, setup_tools
from microbots.utils.logger import LogLevelEmoji, LogTextColor
from microbots.utils.network import get_free_port
from microbots.utils.path import get_path_info

logger = getLogger(" MicroBot ")

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


class BotType(StrEnum):
    READING_BOT = "READING_BOT"
    WRITING_BOT = "WRITING_BOT"
    BROWSING_BOT = "BROWSING_BOT"
    CUSTOM_BOT = "CUSTOM_BOT"
    LOG_ANALYSIS_BOT = "LOG_ANALYSIS_BOT"


@dataclass
class BotRunResult:
    status: bool
    result: str | None
    error: Optional[str]


class MicroBot:

    def __init__(
        self,
        bot_type: BotType,
        model: str,
        system_prompt: Optional[str] = None,
        environment: Optional[any] = None,
        additional_tools: Optional[list[Tool]] = [],
        folder_to_mount: Optional[str] = None,
        permission: Optional[PermissionLabels] = None,
    ):
        # validate init values before assigning
        self.permission = permission
        if folder_to_mount is not None:
            folder_mount_info = get_path_info(folder_to_mount)
            if folder_mount_info.path_valid is False:
                raise ValueError(
                    f"Invalid folder to mount: {folder_to_mount} resolved to {folder_mount_info.abs_path}"
                )
            else:
                self.folder_to_mount = folder_mount_info.abs_path

        self._validate_model_and_provider(model)
        self.permission_key = PermissionMapping.MAPPING.get(self.permission)
        self.system_prompt = system_prompt
        self.model = model
        self.bot_type = bot_type
        self.model_provider = model.split("/")[0]
        self.deployment_name = model.split("/")[1]
        self.environment = environment
        self.additional_tools = additional_tools
        self.folder_to_mount = folder_to_mount
        self._create_environment(self.folder_to_mount)
        self._create_llm()

        install_tools(self.environment, self.additional_tools)

    def run(self, task, max_iterations=20, timeout_in_seconds=200) -> BotRunResult:

        setup_tools(self.environment, self.additional_tools)

        iteration_count = 1
        # start timer
        start_time = time.time()
        timeout = timeout_in_seconds
        llm_response = self.llm.ask(task)
        return_value = BotRunResult(
            status=False,
            result=None,
            error="Did not complete",
        )
        logger.info("%s TASK STARTED : %s...", LogLevelEmoji.INFO, task[0:15])

        while llm_response.task_done is False:
            logger.info("%s Step-%d %s", "-" * 20, iteration_count, "-" * 20)
            logger.info(
                f" ➡️  LLM tool call : {LogTextColor.OKBLUE}{json.dumps(llm_response.command)}{LogTextColor.ENDC}",
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
            if llm_command_output.stdout:
                logger.info(
                    " ⬅️  Command Execution Output: %s",
                    llm_command_output.stdout,
                )

            # Convert CmdReturn to string for LLM
            if llm_command_output.stdout:
                output_text = llm_command_output.stdout
            elif llm_command_output.stderr:
                output_text = f"COMMUNICATION ERROR: {llm_command_output.stderr}"
            else:
                output_text = "No output received"

            llm_response = self.llm.ask(output_text)

        logger.info("🔚 TASK COMPLETED : %s...", task[0:15])
        return BotRunResult(status=True, result=llm_response.result, error=None)

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

    def __del__(self):
        if self.environment:
            try:
                self.environment.stop()
            except Exception as e:
                logger.error(
                    "%s Error while stopping environment: %s", LogLevelEmoji.ERROR, e
                )
