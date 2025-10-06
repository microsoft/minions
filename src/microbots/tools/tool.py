from dataclasses import dataclass
import os
import platform
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
    """
    Tool definition for environment setup and configuration.

    Supports platform-specific commands and versioning for better compatibility
    across different operating systems and environments.
    """
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

    # Tool version (semantic versioning recommended, e.g., "1.0.0")
    version: Optional[str] = None

    # Platform-specific install commands (overrides install_commands if platform matches)
    # Keys: "linux", "darwin" (macOS), "windows", or specific like "ubuntu", "debian", "alpine"
    # Example: {"ubuntu": ["apt-get install -y tool"], "alpine": ["apk add tool"]}
    platform_install_commands: Optional[dict[str, List[str]]] = None

    # Platform-specific verify commands (overrides verify_commands if platform matches)
    platform_verify_commands: Optional[dict[str, List[str]]] = None

    # Platform-specific setup commands (overrides setup_commands if platform matches)
    platform_setup_commands: Optional[dict[str, List[str]]] = None

    # Platform-specific uninstall commands (overrides uninstall_commands if platform matches)
    platform_uninstall_commands: Optional[dict[str, List[str]]] = None

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


def _detect_platform(env: Environment) -> str:
    """
    Detect the platform/OS of the environment.

    Returns a platform identifier like "ubuntu", "debian", "alpine", "linux", "darwin", "windows".
    Tries to detect specific Linux distributions first, then falls back to general platform.
    """
    # Try to detect Linux distribution
    try:
        result = env.execute("cat /etc/os-release 2>/dev/null || echo ''")
        if result.return_code == 0 and result.stdout:
            output_lower = result.stdout.lower()
            if "ubuntu" in output_lower:
                return "ubuntu"
            elif "debian" in output_lower:
                return "debian"
            elif "alpine" in output_lower:
                return "alpine"
            elif "centos" in output_lower or "rhel" in output_lower:
                return "rhel"
            elif "fedora" in output_lower:
                return "fedora"
    except Exception as e:
        logger.debug("Could not detect Linux distribution: %s", e)

    # Fall back to general platform detection
    try:
        result = env.execute("uname -s")
        if result.return_code == 0:
            system = result.stdout.strip().lower()
            if "linux" in system:
                return "linux"
            elif "darwin" in system:
                return "darwin"
            elif "windows" in system or "mingw" in system or "cygwin" in system:
                return "windows"
    except Exception as e:
        logger.debug("Could not detect platform: %s", e)

    # Default to linux (most containers are Linux)
    logger.warning("Could not detect platform, defaulting to 'linux'")
    return "linux"


def _get_platform_specific_commands(
    tool: Tool,
    platform_type: str,
    command_type: str
) -> Optional[List[str]]:
    """
    Get platform-specific commands for a tool, falling back to default if not available.

    Args:
        tool: The Tool instance
        platform_type: Detected platform (e.g., "ubuntu", "linux", "darwin")
        command_type: Type of command ("install", "verify", "setup", "uninstall")

    Returns:
        List of commands to execute, or None if no commands defined
    """
    platform_attr = f"platform_{command_type}_commands"
    default_attr = f"{command_type}_commands"

    # Try platform-specific commands first
    platform_commands = getattr(tool, platform_attr, None)
    if platform_commands and isinstance(platform_commands, dict):
        # Try exact platform match first
        if platform_type in platform_commands:
            logger.debug("Using %s-specific %s commands", platform_type, command_type)
            return platform_commands[platform_type]

        # Try fallback to broader platform (e.g., ubuntu -> linux)
        if platform_type in ["ubuntu", "debian", "alpine", "rhel", "fedora"]:
            if "linux" in platform_commands:
                logger.debug("Using linux fallback %s commands for %s", command_type, platform_type)
                return platform_commands["linux"]

    # Fall back to default commands
    default_commands = getattr(tool, default_attr, None)
    if default_commands:
        logger.debug("Using default %s commands", command_type)
        return default_commands

    return None


def _install_tool(env: Environment, tool: Tool, platform_type: str):
    """Install a tool using platform-specific or default commands."""
    version_info = f" (v{tool.version})" if tool.version else ""
    logger.debug("Installing tool: %s%s", tool.name, version_info)

    install_commands = _get_platform_specific_commands(tool, platform_type, "install")
    if not install_commands:
        logger.warning("No install commands found for tool: %s", tool.name)
        return

    for command in install_commands:
        output = env.execute(command)
        logger.debug("Tool install command: %s", command)
        logger.debug("Tool install command output: %s", output)
        if output.return_code != 0:
            logger.error(
                "❌ Failed to install tool: %s with command: %s",
                tool.name,
                command,
            )
            raise RuntimeError(
                f"Failed to install tool {tool.name}{version_info} with command {command}. Output: {output}"
            )
    logger.info("✅ Successfully installed tool: %s%s", tool.name, version_info)

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

def _verify_tool_installation(env: Environment, tool: Tool, platform_type: str):
    """Verify tool installation using platform-specific or default commands."""
    verify_commands = _get_platform_specific_commands(tool, platform_type, "verify")
    if not verify_commands:
        logger.debug("No verify commands provided for tool: %s", tool.name)
        return

    for command in verify_commands:
        output = env.execute(command)
        logger.debug("Tool verify command: %s", command)
        logger.debug("Tool verify output: %s", output)
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
    """Install and verify tools with platform detection and version support."""
    if not tools:
        return

    # Detect platform once for all tools
    platform_type = _detect_platform(env)
    logger.info("🔍 Detected platform: %s", platform_type)

    for tool in tools:
        _install_tool(env, tool, platform_type)

        # Copy environment variables if specified
        if tool.env_variables:
            for env_variable in tool.env_variables:
                _copy_env_variable(env, env_variable)

        # Copy files if specified
        if tool.files_to_copy:
            for file_copy in tool.files_to_copy:
                _copy_file(env, file_copy)

        # Verify installation
        _verify_tool_installation(env, tool, platform_type)

def setup_tools(env: Environment, tools: List[Tool]):
    """Setup tools using platform-specific or default commands."""
    if not tools:
        logger.debug("No tools provided for setup.")
        return

    # Detect platform once for all tools
    platform_type = _detect_platform(env)

    for tool in tools:
        setup_commands = _get_platform_specific_commands(tool, platform_type, "setup")
        if not setup_commands:
            logger.debug("No setup commands provided for tool: %s", tool.name)
            continue

        env.execute(f"cd /{DOCKER_WORKING_DIR}")

        for command in setup_commands:
            output = env.execute(command)
            logger.debug("Tool setup command: %s", command)
            logger.debug("Tool setup output: %s", output)
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
