from dataclasses import dataclass
import os
from typing import Optional, List
from pathlib import Path
import yaml
import logging

from microbots.environment.Environment import Environment

from microbots.constants import DOCKER_WORKING_DIR, TOOL_FILE_BASE_PATH

logger = logging.getLogger(" 🔧 Tool")


@dataclass
class EnvFileCopies:
    src: Path
    dest: Path
    permissions: int  # Use FILE_PERMISSION enum to set permissions


@dataclass
class Tool:
    # TODO: Handle different instructions based on the platform (linux flavours, windows, mac)
    # TODO: Add versioning to tools
    name: str
    description: str
    parameters: dict | None

    # This is the set of instructions that will be provided to the LLM on how to use this tool.
    # This string will be appended to the LLM's system prompt.
    # This instructions should be non-interactive
    usage_instructions_to_llm: str

    # This set of commands will be executed once the environment is up and running.
    # These commands will be executed in the order they are provided.
    install_commands: List[str]

    # Mention what are the environment variables that need to be copied from your current environment
    env_variables: Optional[str] = None

    # Any files to be copied to the environment before the tool is installed.
    files_to_copy: Optional[List[EnvFileCopies]] = None

    # This set of commands will be executed to verify if the tool is installed correctly.
    # If any of these commands fail, the tool installation is considered to have failed.
    verify_commands: Optional[List[str]] = None

    # This set of commands will be executed after the code is copied to the environment
    # and before the llm is invoked.
    # These commands will be executed inside the mounted folder.
    setup_commands: Optional[List[str]] = None

    # This set of commands will be executed when the environment is being torn down.
    uninstall_commands: Optional[List[str]] = None


def parse_tool_definition(yaml_path: str) -> Tool:
    """
    Parse a tool definition from a YAML file.

    Args:
        yaml_path: The path to the YAML file containing the tool definition.
                   If it is not an absolute path, it is relative to project_root/tool/tool_definition/

    Returns:
        A Tool object parsed from the YAML file.
    """

    yaml_path = Path(yaml_path)

    if not yaml_path.is_absolute():
        yaml_path = Path(__file__).parent / "tool_definitions" / yaml_path

    with open(yaml_path, "r") as f:
        tool_dict = yaml.safe_load(f)

    for file_to_copy in tool_dict.get("files_to_copy", []) or []:
        file_to_copy["src"] = Path(file_to_copy["src"])
        file_to_copy["dest"] = Path(file_to_copy["dest"])
        if "permissions" not in file_to_copy:
            raise ValueError(f"permissions not provided for file copy {file_to_copy}")
        if not isinstance(file_to_copy["permissions"], int) or not (0 <= file_to_copy["permissions"] <= 7):
            raise ValueError(f"permissions must be an integer between 0 and 7 for file copy {file_to_copy}")
        file_to_copy["permissions"] = file_to_copy.pop("permissions")

    tool_dict["files_to_copy"] = [EnvFileCopies(**file_to_copy) for file_to_copy in tool_dict.get("files_to_copy", []) or []]

    return Tool(**tool_dict)


def _install_tool(env: Environment, tool: Tool):
    logger.debug("Installing tool: %s", tool.name)
    for command in tool.install_commands:
        output = env.execute(command)
        logger.debug("Tool install command: %s", output)
        logger.debug("Tool install command output: %s", output)
        if output.return_code != 0:
            logger.error(
                "❌ Failed to install tool: %s with command: %s",
                tool.name,
                command,
            )
            raise RuntimeError(
                f"Failed to install tool {tool.name} with command {command}. Output: {output}"
            )
    logger.info("✅ Successfully installed tool: %s", tool.name)

def _copy_env_variable(env: Environment, env_variable: str):
    if env_variable not in os.environ:
        logger.error(
            "❌ Environment variable %s not found in current environment",
            env_variable,
        )
        raise ValueError(
            f"Environment variable {env_variable} not found in current environment"
        )

    env.execute(
        f'export {env_variable}="{os.environ.get(env_variable)}"'
    )
    logger.info("✅ Set environment variable %s in the container", env_variable)

def _copy_file(env: Environment, file_copy: EnvFileCopies):
    # If not abs path, append to TOOL_FILE_BASE_PATH
    if not file_copy.src.is_absolute():
        file_copy.src = str(TOOL_FILE_BASE_PATH / file_copy.src)

    # We con't have copy functionality yet. Read source file and write to dest
    if not os.path.exists(file_copy.src):
        logger.error(
            "❌ File to copy %s not found in current environment",
            file_copy.src,
        )
        raise ValueError(
            f"File to copy {file_copy.src} not found in current environment"
        )

    with open(file_copy.src, "r") as src_file:
        content = src_file.read()
        # escape all quotes in content
        content = content.replace('"', '\\"')
        # escape backslashes for shell execution
        # content = content.replace('\\', '\\\\')
    dest_path_in_container = f"/{file_copy.dest}"
    output = env.execute(
        f'echo """{content}""" > {dest_path_in_container}'
    )
    if output.return_code != 0:
        logger.error(
            "❌ Failed to copy file to container: %s to: %s",
            file_copy.src,
            dest_path_in_container,
        )
        raise RuntimeError(
            f"Failed to copy file to container {file_copy.dest}. Output: {output}"
        )
    _setup_file_permission(env, file_copy)
    logger.info("✅ Copied file to container: %s to: %s", file_copy.src, dest_path_in_container)

def _setup_file_permission(env: Environment, file_copy: EnvFileCopies):
    permission_command = ""
    if file_copy.permissions - 4 >= 0:
        permission_command += f"chmod +r {file_copy.dest} && "
    if file_copy.permissions - 2 >= 0:
        permission_command += f"chmod +w {file_copy.dest} && "
    if file_copy.permissions - 1 >= 0:
        permission_command += f"chmod +x {file_copy.dest}"
    output = env.execute(permission_command)
    if output.return_code != 0:
        logger.error(
            "❌ Failed to set permission for file in container: %s to: %s",
            file_copy.src,
            file_copy.dest,
        )
        raise RuntimeError(
            f"Failed to set permission for file in container {file_copy.dest}. Output: {output}"
        )

def _verify_tool_installation(env: Environment, tool: Tool):
    if not tool.verify_commands:
        logger.debug("No verify commands provided for tool: %s", tool.name)
        return

    for command in tool.verify_commands:
        output = env.execute(command)
        logger.debug("Tool verify command: %s", output)
        if output.return_code != 0:
            logger.error(
                "❌ Failed to verify tool: %s with command: %s",
                tool.name,
                command,
            )
            raise RuntimeError(
                f"Failed to verify tool {tool.name} with command {command}. Output: {output}"
            )
    logger.info("✅ Successfully installed and verified tool: %s", tool.name)

def install_tools(env: Environment, tools: List[Tool]):
    if tools:
        for tool in tools:
            _install_tool(env, tool)

        for env_variable in tool.env_variables:
            _copy_env_variable(env, env_variable)

        for file_copy in tool.files_to_copy or []:
            _copy_file(env, file_copy)

        for tool in tools:
            _verify_tool_installation(env, tool)

def setup_tools(env: Environment, tools: List[Tool]):
    if not tools:
        logger.debug("No tools provided for setup.")
        return

    for tool in tools:
        if not tool.setup_commands:
            logger.debug("No setup commands provided for tool: %s", tool.name)
            continue

        env.execute(f"cd /{DOCKER_WORKING_DIR}")

        for command in tool.setup_commands:
            output = env.execute(command)
            logger.debug("Tool setup command: %s", output)
            if output.return_code != 0:
                logger.error(
                    "❌ Failed to setup tool: %s with command: %s",
                    tool.name,
                    command,
                )
                raise RuntimeError(
                    f"Failed to setup tool {tool.name} with command {command}. Output: {output}"
                )
    logger.info("✅ Successfully setup tool: %s", tool.name)
