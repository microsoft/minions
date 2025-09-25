from enum import IntEnum, StrEnum
from pathlib import Path


class ModelProvider(StrEnum):
    OPENAI = "azure-openai"


class ModelEnum(StrEnum):
    GPT_5 = "gpt-5"


class PermissionLabels(StrEnum):
    READ_ONLY = "READ_ONLY"
    READ_WRITE = "READ_WRITE"


class PermissionMapping:
    MAPPING = {
        PermissionLabels.READ_ONLY: "ro",
        PermissionLabels.READ_WRITE: "rw",
    }


class FILE_PERMISSIONS(IntEnum):
    READ = 4
    WRITE = 2
    EXECUTE = 1


DOCKER_WORKING_DIR = "workdir"
LOG_FILE_DIR = "/var/log/"
TOOL_FILE_BASE_PATH = Path(__file__).parent / "tools" / "tool_definitions"
