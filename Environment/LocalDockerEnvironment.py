import docker
from typing import Optional, Dict, Any
import time
import os
import requests

class LocalDockerEnvironment:
    def __init__(self, folder_to_mount: str, permission: str):
        if permission not in ["READ_ONLY", "READ_WRITE"]:
            raise ValueError("permission must be 'READ_ONLY' or 'READ_WRITE'")

        self.image = "shell_server:latest"  # rename if needed
        self.folder_to_mount = folder_to_mount
        self.permission = permission
        self.container = None
        self.client = docker.from_env()
        self.port = 8080  # Default port; can be parameterized if needed
        self.start()

    def start(self):
        mode_map = {
        "READ_ONLY": "ro",
        "READ_WRITE": "rw"
        }
        volumes_config = {}
        if self.folder_to_mount:
            volumes_config[self.folder_to_mount] = {
                "bind": f"/app/{os.path.basename(self.folder_to_mount)}",
                "mode": mode_map[self.permission],
            }
            print(f"Volume mapping: {self.folder_to_mount} -> /app/{os.path.basename(self.folder_to_mount)}")

        self.container = self.client.containers.run(
            self.image,
            volumes=volumes_config,
            ports={f'{self.port}/tcp': self.port},
            detach=True,
            working_dir="/app",
        )
        print(f"Started container {self.container.id[:12]} with image {self.image}")
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
        except requests.exceptions.ConnectionError as e:
            print("Checking container status...")
            self.container.reload()
            print(f"Container status: {self.container.status}")
            if self.container.status != "running":
                print("Container logs:")
                print(self.container.logs().decode("utf-8"))
            return f"Error: Could not connect"
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return f"Error: Request failed"

