from enum import Enum
from typing import Optional


class ModelProvider(Enum):
    OPENAI = "openai"


class ModelEnum(Enum):
    GPT_5 = "gpt-5"


class PermissionLabels(Enum):
    READ_ONLY = "READ_ONLY"
    READ_WRITE = "READ_WRITE"


class PermissionMapping:
    MAPPING = {
        PermissionLabels.READ_ONLY: "ro",
        PermissionLabels.READ_WRITE: "rw",
    }


class AgentType(Enum):
    READING_AGENT = "READING_AGENT"
    WRITING_AGENT = "WRITING_AGENT"
    BROWSING_AGENT = "BROWSING_AGENT"
    CUSTOM_AGENT = "CUSTOM_AGENT"


class AgentRunResult(Enum):
    status: bool
    result: str | None
    error: Optional[str]


llm_output_format = """```json
{
    task_done: true | false, 
    command: "<command to run> | null", 
    result: str | null
}
```
"""


def get_system_prompt(
    agent_type: AgentType, mounted_directory: str, system_prompt: str | None
) -> str:

    # Create system prompt based on agent_type and permission mapping
    system_prompt_common = """There is a shell session open for you. 
            I will provide a task to achieve using the shell. 
            You will provide the commands to achieve the task in this particular below json format 
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
    You are only provided access to read and write files inside the mounted directory {mounted_directory}.
    You are also provided access to internet to search for information.
    """,
        AgentType.CUSTOM_AGENT: system_prompt,
    }

    return system_prompts.get(agent_type)


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

    elif agent_type == AgentType.WRITING_AGENT and folder_to_mount is not None:
        raise ValueError("Folder to mount should provided for writing agents")

    elif agent_type == AgentType.READING_AGENT and folder_to_mount is not None:
        raise ValueError("Folder to mount should provided for reading agents")

    elif agent_type == AgentType.BROWSING_AGENT and folder_to_mount is not None:
        raise ValueError("Folder to mount should not be provided for browsing agents")
