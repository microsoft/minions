from abc import ABC, abstractmethod


class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def installation_command(self) -> str:
        pass

    @property
    @abstractmethod
    def verification_command(self) -> str:
        pass

    @property
    @abstractmethod
    def usage_instructions_to_llm(self) -> str:
        pass
