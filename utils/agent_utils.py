import logging
import socket

from tool_definitions.base_tool import BaseTool
from tool_definitions.ctags import Ctags
from tool_definitions.node import Node
from types_and_constants.agent import AgentType, PermissionLabels

logger = logging.getLogger(__name__)


def get_default_tools() -> list[BaseTool]:
    return []


llm_output_format = """```json
{
    task_done: true | false, 
    command: "<command to run> | null", 
    result: str | null
}
```
"""


def get_free_port():
    # Create a temporary socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Bind the socket to address 0.0.0.0 and port 0
        # Port 0 tells the OS to assign a free ephemeral port
        sock.bind(("0.0.0.0", 0))
        # Get the port number that was assigned
        port = sock.getsockname()[1]
        return port
    finally:
        # Close the socket to release the port
        sock.close()


def get_system_prompt(
    agent_type: AgentType, mounted_directory: str, system_prompt: str | None
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


def assign_permission_based_on_agent_type(agent_type, permission):
    return_value = permission
    if permission is None and agent_type == AgentType.CUSTOM_AGENT:
        return_value = PermissionLabels.READ_WRITE
    else:
        if agent_type == AgentType.READING_AGENT:
            return_value = PermissionLabels.READ_ONLY
        elif agent_type == AgentType.WRITING_AGENT:
            return_value = PermissionLabels.READ_WRITE
    return return_value


def validate_agent_type_and_system_prompt(agent_type, system_prompt):
    if agent_type == AgentType.CUSTOM_AGENT and system_prompt is None:
        raise ValueError("Custom agent requires a system prompt")

    elif (
        agent_type == AgentType.READING_AGENT
        or agent_type == AgentType.BROWSING_AGENT
        or agent_type == AgentType.WRITING_AGENT
    ) and system_prompt is not None:
        raise ValueError("System prompt should not be provided for non-custom agents")


def validate_agent_type_and_permission(agent_type, permission):
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


def validate_agent_type_and_folder_to_mount(agent_type, folder_to_mount):

    if agent_type == AgentType.CUSTOM_AGENT and folder_to_mount is None:
        raise ValueError("Folder to mount is required for custom agent")

    elif agent_type == AgentType.WRITING_AGENT and folder_to_mount is None:
        raise ValueError("Folder to mount should provided for writing agents")

    elif agent_type == AgentType.READING_AGENT and folder_to_mount is None:
        raise ValueError("Folder to mount should provided for reading agents")

    elif agent_type == AgentType.BROWSING_AGENT and folder_to_mount is not None:
        raise ValueError("Folder to mount should not be provided for browsing agents")
