import json
import time
from dataclasses import dataclass
from enum import StrEnum
from logging import getLogger
from typing import Optional

from microbots.constants import ModelProvider
from microbots.environment.local_docker.LocalDockerEnvironment import (
    LocalDockerEnvironment,
)
from microbots.llm.openai_api import OpenAIApi
from microbots.llm.llm import llm_output_format_str
from microbots.tools.tool import Tool, install_tools, setup_tools
from microbots.extras.mount import Mount, MountType
from microbots.utils.logger import LogLevelEmoji, LogTextColor
from microbots.utils.network import get_free_port

logger = getLogger(" MicroBot ")

system_prompt_common = f"""There is a shell session open for you.
                I will provide a task to achieve using the shell.
                You will provide the commands to achieve the task in this particular below json format, Ensure all the time to respond in this format only and nothing else, also all the properties ( task_done, command, result ) are mandatory on each response
                {llm_output_format_str}
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
    """
    The core Microbot class.

    MicroBot class is the core class representing the autonomous agent. Other bots are extensions of this class.
    If you want to create a custom bot, you can directly use this class or extend it into your own bot class.

    Attributes
    ----------
        model : str
            The model to use for the bot, in the format <provider>/<model_name>.
        bot_type : BotType
            The type of bot being created. It's unused. Will be removed soon.
        system_prompt : Optional[str]
            The system prompt to guide the bot's behavior.
        environment : Optional[any]
            The execution environment for the bot. If not provided, a default
            LocalDockerEnvironment will be created.
        additional_tools : Optional[list[Tool]]
            A list of additional tools to install in the bot's environment.
        folder_to_mount : Optional[Mount]
            A folder to mount into the bot's environment. The bot will be given
            access to this folder based on the specified permissions. This will
            be the main code folder where the bot will work. Additional folders
            can be mounted during the run() method. Refer to `Mount` class
            regarding the directory structure and permission details. Defaults
            to None.
    """

    def __init__(
        self,
        model: str,
        bot_type: BotType = BotType.CUSTOM_BOT,
        system_prompt: Optional[str] = None,
        environment: Optional[any] = None,
        additional_tools: Optional[list[Tool]] = [],
        folder_to_mount: Optional[Mount] = None,
    ):
        """
        Init function for MicroBot class.

        Parameters
        ----------
            model :str
                The model to use for the bot, in the format <provider>/<model_name>.
            bot_type :BotType
                The type of bot being created. It's unused. Will be removed soon.
            system_prompt :Optional[str]
                The system prompt to guide the bot's behavior. Defaults to None.
            environment :Optional[any]
                The execution environment for the bot. If not provided, a default
                LocalDockerEnvironment will be created.
            additional_tools :Optional[list[Tool]]
                A list of additional tools to install in the bot's environment.
                Defaults to [].
            folder_to_mount :Optional[Mount]
                A folder to mount into the bot's environment. The bot will be given
                access to this folder based on the specified permissions. This will
                be the main code folder where the bot will work. Additional folders
                can be mounted using the run() method. Refer to `Mount` class
                regarding the directory structure and permission details. Defaults
                to None.

                Note: Supports only mount type MountType.MOUNT for now.
        """

        self.folder_to_mount = folder_to_mount

        # TODO : Need to check on the purpose of variable `mounted`
        # 1. If we allow user to mount multiple directories,
        # we should able to get it as an argument and store them in self.mounted.
        # This require changes in _create_environment to handle multiple mount directories or files.
        #
        # 2. We should let user to mount only one directory. In that case self.mounted may not be required.
        # Just one self.folder_to_mount and necessary extra mounts at the derived class similar to LogAnalyticsBot.

        self.mounted = []
        if folder_to_mount is not None:
            self._validate_folder_to_mount(folder_to_mount)
            self.mounted.append(folder_to_mount)

        self.system_prompt = system_prompt
        self.model = model
        self.bot_type = bot_type
        self.environment = environment
        self.additional_tools = additional_tools

        self._validate_model_and_provider(model)
        self.model_provider = model.split("/")[0]
        self.deployment_name = model.split("/")[1]

        if not self.environment:
            self._create_environment(self.folder_to_mount)

        self._create_llm()

        install_tools(self.environment, self.additional_tools)

    def run(
        self,
        task: str,
        additional_mounts: Optional[list[Mount]] = None,
        max_iterations: int = 20,
        timeout_in_seconds: int = 200
    ) -> BotRunResult:

        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than 0")

        setup_tools(self.environment, self.additional_tools)

        for mount in additional_mounts or []:
            self._mount_additional(mount)

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
        logger.info("%s TASK STARTED : %s...", LogLevelEmoji.INFO, task[0:60])

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
                    "Iteration %d with response %s - Exiting without running command as timeout reached",
                    iteration_count,
                    llm_response,
                )
                return_value.error = f"Timeout of {timeout} seconds reached"
                return return_value

            llm_command_output = self.environment.execute(llm_response.command)

            logger.info(
                    " ⬅️  Command executed.\nReturn Code: %d\nStdout:\n%s\nStderr:\n%s",
                    llm_command_output.return_code,
                    llm_command_output.stdout,
                    llm_command_output.stderr,
                )

            if llm_command_output.return_code == 0:
                if llm_command_output.stdout:
                    output_text = llm_command_output.stdout
                else:
                    output_text = f"Command executed successfully with no output\nreturn code: {llm_command_output.return_code}\nstdout: {llm_command_output.stdout}\nstderr: {llm_command_output.stderr}"
            else:
                output_text = f"COMMAND EXECUTION FAILED\nreturn code: {llm_command_output.return_code}\nstdout: {llm_command_output.stdout}\nstderr: {llm_command_output.stderr}"

            logger.info("📨  Message to LLM:\n%s", output_text)
            llm_response = self.llm.ask(output_text)

        logger.info("🔚 TASK COMPLETED : %s...", task[0:15])
        return BotRunResult(status=True, result=llm_response.result, error=None)

    def _mount_additional(self, mount: Mount):
        if mount.mount_type != MountType.COPY:
            logger.error(
                "%s Only COPY mount type is supported for additional mounts for now",
                LogLevelEmoji.ERROR,
            )
            raise ValueError(
                "Only COPY mount type is supported for additional mounts for now"
            )

        self.mounted.append(mount)
        copy_to_container_result = self.environment.copy_to_container(
            mount.host_path_info.abs_path, mount.sandbox_path
        )
        if copy_to_container_result is False:
            raise ValueError(
                f"Failed to copy additional mount to container: {mount.host_path_info.abs_path} -> {mount.sandbox_path}"
            )

    # TODO : pass the sandbox path
    def _create_environment(self, folder_to_mount: Optional[Mount]):
        free_port = get_free_port()

        self.environment = LocalDockerEnvironment(
            port=free_port,
            folder_to_mount=folder_to_mount,
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

    def _validate_folder_to_mount(self, folder_to_mount: Mount):
        if folder_to_mount.mount_type != MountType.MOUNT:
            logger.error(
                "%s Only MOUNT mount type is supported for folder_to_mount",
                LogLevelEmoji.ERROR,
            )
            raise ValueError(
                "Only MOUNT mount type is supported for folder_to_mount"
            )
