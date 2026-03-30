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

    def expose_port(self, container_port: int, host_port: int) -> bool:
        """Expose an additional port from the running environment.

        Makes a service listening on *container_port* inside the environment
        reachable at *host_port* on the host.  How this is achieved is up to
        the implementation (e.g. socat, iptables, native platform API).

        Parameters
        ----------
        container_port : int
            The port the service is listening on **inside** the environment.
        host_port : int
            The port on the **host** that should forward to *container_port*.

        Returns
        -------
        bool
            True if the port was exposed successfully, False otherwise.

        Raises
        ------
        NotImplementedError
            If the environment does not support dynamic port exposure.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support exposing additional ports. "
            f"This is an optional feature - only implement if needed for your use case."
        )
