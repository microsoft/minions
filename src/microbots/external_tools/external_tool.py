from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

from microbots.MicroBot import MicroBot


logger = logging.getLogger(" 🔧 External Tool ")

@dataclass
class ExternalTool(ABC):
    name: str
    command: str
    description: str
    usage_instructions_to_llm: str

    @abstractmethod
    def call(self, microbot: MicroBot, command: str) -> str:
        pass