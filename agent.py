import json
import os
import time
from logging import getLogger
from typing import Optional

from Environment.LocalDockerEnvironment import LocalDockerEnvironment
from llm.openai_api import OpenAIApi
from tool_definitions.base_tool import BaseTool
from types_and_constants.agent import (
    AgentRunResult,
    AgentType,
    ModelProvider,
    PermissionLabels,
    PermissionMapping,
)
from utils.agent_utils import (
    assign_permission_based_on_agent_type,
    get_default_tools,
    get_system_prompt,
    validate_agent_type_and_folder_to_mount,
    validate_agent_type_and_permission,
    validate_agent_type_and_system_prompt,
    validate_model_and_provider,
)
from utils.logger import LogLevelEmoji
from utils.network import get_free_port

logger = getLogger(__name__)


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
        self.permission = assign_permission_based_on_agent_type(agent_type, permission)
        self._agent_arg_validation(
            agent_type, system_prompt, self.permission, folder_to_mount, model
        )
        if folder_to_mount is not None:
            self.folder_to_mount_base_path = os.path.basename(folder_to_mount)

        self.permission_key = PermissionMapping.MAPPING.get(self.permission)
        self.system_prompt = get_system_prompt(
            agent_type, self.folder_to_mount_base_path, system_prompt
        )
        self.tools = get_default_tools() + additional_tools
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
        return_value: AgentRunResult = {
            "status": False,
            "error": "Did not complete",
        }
        logger.info("%s TASK STARTED : %s...", LogLevelEmoji.INFO, task[0:15])
        while llm_response["task_done"] is False:
            print(
                "----------------------------------------------------------------------------------------------------------"
            )
            logger.info(
                " %s Task Iteration Count : %d", LogLevelEmoji.INFO, iteration_count
            )
            logger.info(
                " %s After LLM Communication Response : %s",
                LogLevelEmoji.INFO,
                json.dumps(llm_response["command"]),
            )
            # increment iteration count
            iteration_count += 1
            if iteration_count >= max_iterations:
                return_value["error"] = f"Max iterations {max_iterations} reached"
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
                return_value["error"] = f"Timeout of {timeout} seconds reached"
                return return_value

            llm_command_output = self.environment.execute(llm_response["command"])
            llm_response = self.llm.ask(llm_command_output)

        logger.info("%s TASK COMPLETED : %s...", LogLevelEmoji.COMPLETED, task[0:15])
        return {"status": True, "result": llm_response["result"]}

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
        validate_agent_type_and_system_prompt(agent_type, system_prompt)
        validate_agent_type_and_permission(agent_type, permission)
        validate_agent_type_and_folder_to_mount(agent_type, folder_to_mount)
        validate_model_and_provider(model)
