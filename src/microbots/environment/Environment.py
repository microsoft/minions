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

    @abstractmethod
    def copy_to_container(self, src_path: str) -> bool:
        pass

    @abstractmethod
    def copy_from_container(self, src_path: str, dest_path: str) -> bool:
        pass
