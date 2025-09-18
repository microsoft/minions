from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass

@dataclass
class CmdReturn:
    stdout: str
    stderr: str
    return_code: int


class Environment(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def execute(self, command: str, timeout: Optional[int] = 300) -> CmdReturn:
        pass
