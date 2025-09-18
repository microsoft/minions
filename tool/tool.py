from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import yaml

@dataclass
class Tool:
    # TODO: Handle different instructions based on the platform (linux flavours, windows, mac)
    # TODO: Add versioning to tools
    name: Optional[str]
    description: Optional[str]

    # This is the set of instructions that will be provided to the LLM on how to use this tool.
    # This string will be appended to the LLM's system prompt.
    # This instructions should be non-interactive
    usage_instructions_to_llm: str

    # This set of commands will be executed once the environment is up and running.
    # These commands will be executed in the order they are provided.
    install_commands: List[str]

    # This set of commands will be executed to verify if the tool is installed correctly.
    # If any of these commands fail, the tool installation is considered to have failed.
    verify_commands: Optional[List[str]] = None

    # This set of commands will be executed after the code is copied to the environment
    # and before the llm is invoked.
    # These commands will be executed inside the mounted folder.
    setup_commands: Optional[List[str]] = None

    # This set of commands will be executed when the environment is being torn down.
    uninstall_commands: Optional[List[str]] = None


def parse_tool_definition(yaml_path: Path) -> Tool:
    """
    Parse a tool definition from a YAML file.

    Args:
        yaml_path: The path to the YAML file containing the tool definition.
                   If it is not an absolute path, it is relative to project_root/tool/tool_definition/

    Returns:
        A Tool object parsed from the YAML file.
    """

    if not yaml_path.is_absolute():
        yaml_path = Path(__file__).parent / "tool_definition" / yaml_path

    with open(yaml_path, "r") as f:
        tool_dict = yaml.safe_load(f)
    return Tool(**tool_dict)
