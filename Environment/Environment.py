from abc import ABC, abstractmethod
from typing import Optional

class CmdReturn:
    def __init__(self, stdout: str, stderr: str, return_code: int):
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code


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