from enum import Enum, StrEnum
from typing import Optional, TypedDict


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
