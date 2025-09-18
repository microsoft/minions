from enum import StrEnum, IntEnum
from pathlib import Path


class ModelProvider(StrEnum):
    OPENAI = "openai"


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


class FILE_PERMISSION(IntEnum):
    READ = 4
    WRITE = 2
    EXECUTE = 1


DOCKER_WORKING_DIR = "workdir"
TOOL_FILE_BASE_PATH = Path(__file__).parent.resolve() / "tools/tool_definitions"
