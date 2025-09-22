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

    def copy_to_container(self, src_path: str, dest_path: str) -> bool:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support copying files to container. "
            f"This is an optional feature - only implement if needed for your use case."
        )

    def copy_from_container(self, src_path: str, dest_path: str) -> bool:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support copying files from container. "
            f"This is an optional feature - only implement if needed for your use case."
        )
