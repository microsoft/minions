"""
MicroBot: Base class for containerized LLM-driven automation.

This module provides the core MicroBot implementation that creates isolated Docker
environments for executing LLM-driven tasks with controlled file system access.
"""

import json
import time
from dataclasses import dataclass
from enum import StrEnum
from logging import getLogger
from typing import Any, Optional

from microbots.constants import DOCKER_WORKING_DIR, ModelProvider
from microbots.environment.local_docker.LocalDockerEnvironment import (
    LocalDockerEnvironment,
)
from microbots.llm.openai_api import OpenAIApi
from microbots.tools.tool import Tool, install_tools, setup_tools
from microbots.utils.env_mount import Mount
from microbots.utils.logger import LogLevelEmoji, LogTextColor
from microbots.utils.network import get_free_port

logger = getLogger(" MicroBot ")

# LLM Response Format Template
# The LLM must respond with this exact JSON structure for each interaction
LLM_OUTPUT_FORMAT = """```json
{
    "task_done": true | false,
    "command": "<shell command to execute>" | null,
    "result": "final result string" | null
}
```"""

# Base System Prompt for All Bots
# This prompt establishes the interaction protocol between the LLM and environment
SYSTEM_PROMPT_BASE = f"""You have access to a shell session for executing commands.

Your task is to complete the given objective using the available shell commands.

RESPONSE FORMAT (required for every response):
{LLM_OUTPUT_FORMAT}

IMPORTANT RULES:
- All three properties (task_done, command, result) are MANDATORY in every response
- Execute only ONE command at a time
- After each command, you will receive its output
- When the task is complete, set task_done=true and provide the result
- You cannot ask for clarification once the task begins

The system will execute your commands and return their output for the next iteration."""


class BotType(StrEnum):
    """Types of bots available in the MicroBots framework."""

    READING_BOT = "READING_BOT"
    WRITING_BOT = "WRITING_BOT"
    BROWSING_BOT = "BROWSING_BOT"
    CUSTOM_BOT = "CUSTOM_BOT"
    LOG_ANALYSIS_BOT = "LOG_ANALYSIS_BOT"


@dataclass
class BotRunResult:
    """
    Result of a bot execution run.

    Attributes:
        status: True if task completed successfully, False otherwise
        result: The final result string from the LLM, or None if incomplete
        error: Error message if execution failed, None otherwise
    """

    status: bool
    result: str | None
    error: Optional[str]


class MicroBot:
    """
    Base class for all MicroBot implementations.

    MicroBot creates a containerized environment for executing LLM-driven tasks
    with controlled file system access and tool availability.

    Args:
        bot_type: The type of bot (READING_BOT, WRITING_BOT, etc.)
        model: Model string in format "<provider>/<deployment_name>"
        system_prompt: Custom system prompt for the LLM
        environment: Custom environment instance (defaults to LocalDockerEnvironment)
        additional_tools: List of Tool instances to install in the environment
        folder_to_mount: Primary mount configuration (passed to environment at creation)
        additional_mounts: Additional mount configurations (tracked for reference,
                          files can be accessed via copy_to_container method)

    Note:
        Docker containers require volume mounts at creation time. The primary folder_to_mount
        is mounted as a Docker volume. Additional mounts are tracked but require using
        copy_to_container() to access their files within the container.
    """

    def __init__(
        self,
        bot_type: BotType,
        model: str,
        system_prompt: Optional[str] = None,
        environment: Optional[Any] = None,
        additional_tools: Optional[list[Tool]] = None,
        folder_to_mount: Optional[Mount] = None,
        additional_mounts: Optional[list[Mount]] = None,
    ):
        if additional_tools is None:
            additional_tools = []
        if additional_mounts is None:
            additional_mounts = []

        self.folder_to_mount = folder_to_mount
        self.mounted = self._initialize_mounts(folder_to_mount, additional_mounts)

        self._validate_model_and_provider(model)
        self.system_prompt = system_prompt
        self.model = model
        self.bot_type = bot_type
        self.model_provider = model.split("/")[0]
        self.deployment_name = model.split("/")[1]
        self.environment = environment
        self.additional_tools = additional_tools

        self._create_environment(self.folder_to_mount)
        self._create_llm()
        install_tools(self.environment, self.additional_tools)

    def _initialize_mounts(
        self,
        folder_to_mount: Optional[Mount],
        additional_mounts: list[Mount],
    ) -> list[Mount]:
        """
        Initialize and track all mount configurations.

        The primary mount is passed to the Docker environment at creation time.
        Additional mounts are tracked for reference and can be copied into the
        container using copy_to_container() method.

        Args:
            folder_to_mount: Primary mount (will be Docker volume)
            additional_mounts: Additional mounts (require copy_to_container)

        Returns:
            List of all Mount configurations
        """
        mounts = []
        if folder_to_mount is not None:
            mounts.append(folder_to_mount)
        if additional_mounts:
            mounts.extend(additional_mounts)
            logger.info(
                "📋 Tracking %d additional mounts (use copy_to_container for file access)",
                len(additional_mounts)
            )
        return mounts

    def run(self, task: str, max_iterations: int = 20, timeout_in_seconds: int = 200) -> BotRunResult:
        """
        Execute a task using the LLM in the containerized environment.

        Args:
            task: The task description for the LLM to complete
            max_iterations: Maximum number of command execution iterations (default: 20)
            timeout_in_seconds: Maximum time in seconds for task completion (default: 200)

        Returns:
            BotRunResult containing status, result, and any error information
        """
        setup_tools(self.environment, self.additional_tools)

        iteration_count = 1
        start_time = time.time()
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

            iteration_count += 1
            if iteration_count >= max_iterations:
                return_value.error = f"Max iterations {max_iterations} reached"
                return return_value

            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_in_seconds:
                logger.error(
                    "Iteration %d with response %s",
                    iteration_count,
                    json.dumps(llm_response),
                )
                return_value.error = f"Timeout of {timeout_in_seconds} seconds reached"
                return return_value

            llm_command_output = self.environment.execute(llm_response.command)
            if llm_command_output.stdout:
                logger.info(" ⬅️  Command Execution Output: %s", llm_command_output.stdout)

            output_text = self._format_command_output(llm_command_output)
            llm_response = self.llm.ask(output_text)

        logger.info("🔚 TASK COMPLETED : %s...", task[0:15])
        return BotRunResult(status=True, result=llm_response.result, error=None)

    def _format_command_output(self, cmd_output) -> str:
        """
        Format command output for LLM consumption.

        Args:
            cmd_output: CmdReturn object from environment.execute()

        Returns:
            Formatted output string
        """
        if cmd_output.stdout:
            return cmd_output.stdout
        elif cmd_output.stderr:
            return f"COMMUNICATION ERROR: {cmd_output.stderr}"
        else:
            return "No output received"

    def copy_additional_mounts_to_container(self) -> bool:
        """
        Copy all additional mounts to the container.

        This is a convenience method to copy files/directories from additional_mounts
        into the container's working directory. Useful when you need to access files
        from multiple source directories.

        Returns:
            bool: True if all copies succeeded, False otherwise
        """
        if not self.mounted:
            logger.debug("No additional mounts to copy")
            return True

        # Skip the primary mount (already mounted as volume)
        additional = self.mounted[1:] if len(self.mounted) > 1 else []

        if not additional:
            logger.debug("No additional mounts beyond primary to copy")
            return True

        success = True
        for mount in additional:
            try:
                dest_path = f"/{DOCKER_WORKING_DIR}/{mount.host_path_info.base_name}"
                logger.info(
                    "📂 Copying additional mount: %s -> %s",
                    mount.host_path_info.abs_path,
                    dest_path
                )
                result = self.environment.copy_to_container(
                    mount.host_path_info.abs_path,
                    dest_path
                )
                if not result:
                    logger.error("Failed to copy mount: %s", mount.host_path_info.abs_path)
                    success = False
            except Exception as e:
                logger.error("Error copying mount %s: %s", mount.host_path_info.abs_path, e)
                success = False

        return success

    def _create_environment(self, folder_to_mount: Optional[Mount]) -> None:
        """
        Create and initialize the Docker environment.

        Args:
            folder_to_mount: Primary mount configuration for the environment
        """
        if self.environment is None:
            free_port = get_free_port()
            self.environment = LocalDockerEnvironment(
                port=free_port,
                folder_to_mount=(
                    folder_to_mount.host_path_info.abs_path if folder_to_mount else None
                ),
                permission=folder_to_mount.permission if folder_to_mount else None,
            )

    def _create_llm(self) -> None:
        """Create and initialize the LLM client based on model provider."""
        if self.model_provider == ModelProvider.OPENAI:
            self.llm = OpenAIApi(
                system_prompt=self.system_prompt, deployment_name=self.deployment_name
            )

    def _validate_model_and_provider(self, model: str) -> None:
        """
        Validate the model string format and provider.

        Args:
            model: Model string in format "<provider>/<model_name>"

        Raises:
            ValueError: If model format is invalid or provider unsupported
        """
        if model.count("/") != 1:
            raise ValueError("Model should be in the format <provider>/<model_name>")

        provider = model.split("/")[0]
        if provider not in [e.value for e in ModelProvider]:
            raise ValueError(f"Unsupported model provider: {provider}")

    def __del__(self) -> None:
        """Clean up resources when bot instance is destroyed."""
        if self.environment:
            try:
                self.environment.stop()
            except Exception as e:
                logger.error(
                    "%s Error while stopping environment: %s", LogLevelEmoji.ERROR, e
                )
