# from dataclasses import dataclass
from abc import abstractmethod, ABC
import logging

from microbots.MicroBot import MicroBot
from microbots.environment.Environment import CmdReturn  # noqa: E402
from microbots.tools.tool import Tool


logger = logging.getLogger(" 🔧 External Tool ")

class ExternalTool(Tool):

    def __init__(self, name, description, usage_instructions_to_llm):
        self.is_external_tool = True
        super().__init__(name, description, usage_instructions_to_llm, install_commands=[])

    @abstractmethod
    def call(self, microbot: MicroBot, command: str) -> CmdReturn:
        pass