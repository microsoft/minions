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


class AgentType(StrEnum):
    READING_AGENT = "READING_AGENT"
    WRITING_AGENT = "WRITING_AGENT"
    BROWSING_AGENT = "BROWSING_AGENT"
    CUSTOM_AGENT = "CUSTOM_AGENT"


class AgentRunResult(TypedDict):
    status: bool
    result: str | None
    error: Optional[str]
