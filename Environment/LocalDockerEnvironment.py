import docker
from typing import Optional, Dict, Any
import time
import os
import requests
import logging

logger = logging.getLogger(__name__)

class LocalDockerEnvironment:
    def __init__(
        self,
        port: int,
        folder_to_mount: Optional[str] = None,
        permission: Optional[str] = None,
        image: str = "shell_server:latest",
        
    ):
        if folder_to_mount is None and permission is not None:
            raise ValueError("permission provided but folder_to_mount is None")
        elif permission is None and folder_to_mount is not None:
            raise ValueError("folder_to_mount provided but permission is None")
        if permission is not None and permission not in ["READ_ONLY", "READ_WRITE"]:
            raise ValueError("permission must be 'READ_ONLY' or 'READ_WRITE' when provided")

        self.image = image
        self.folder_to_mount = folder_to_mount
        self.permission = permission
        self.container = None
        self.client = docker.from_env()
        self.port = port  # required host port
        self.container_port = 8080
        self.start()

    def start(self):
        mode_map = {
        "READ_ONLY": "ro",
        "READ_WRITE": "rw"
        }
        volumes_config = {}
        if self.folder_to_mount and self.permission:
            volumes_config[self.folder_to_mount] = {
                "bind": f"/app/{os.path.basename(self.folder_to_mount)}",
                "mode": mode_map[self.permission],
            }
            logger.debug(
                "Volume mapping: %s -> /app/%s",
                self.folder_to_mount,
                os.path.basename(self.folder_to_mount),
            )

        # Port mapping
        port_mapping = {f"{self.container_port}/tcp": self.port}

        self.container = self.client.containers.run(
            self.image,
            volumes=volumes_config,
            ports=port_mapping,
            detach=True,
            working_dir="/app",
            environment={"AGENT_PORT": str(self.container_port)},
        )
        logger.info(
            "Started container %s with image %s on host port %s",
            self.container.id[:12],
            self.image,
            self.port,
        )
        time.sleep(2) # Give some time for the server to start

    def stop(self):
        """Stop and remove the container"""
        if self.container:
            self.container.stop()
            self.container.remove()
            self.container = None

    def execute(self, command: str, timeout: Optional[int] = 10) -> Dict[str, Any]:
        try:
            response = requests.post(
                f"http://localhost:{self.port}/",
                json={"message": command},
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json().get("output", "")
        except requests.exceptions.ConnectionError:
            logger.warning("Connection error when executing command; checking container status")
            self.container.reload()
            logger.info("Container status: %s", self.container.status)
            if self.container.status != "running":
                logs = self.container.logs().decode("utf-8", errors="replace")
                logger.error("Container not running. Recent logs below:\n%s", logs)
            return f"Error: Could not connect"
        except requests.exceptions.RequestException as e:
            logger.exception("Request failed while executing command: %s", e)
            return f"Error: Request failed"

