import logging
import os
import time
from pathlib import Path
from typing import Optional

from environment.Environment import Environment, CmdReturn

import docker
import requests

logger = logging.getLogger(__name__)

WORKING_DIR = str(Path.home() / "MICROBOT_WORKDIR")
DOCKER_WORKING_DIR = "/workdir"


class LocalDockerEnvironment(Environment):
    def __init__(
        self,
        port: int,
        folder_to_mount: Optional[str] = None,
        permission: Optional[str] = None,
        image: str = "kavyasree261002/shell_server:latest",
    ):
        if folder_to_mount is None and permission is not None:
            raise ValueError("permission provided but folder_to_mount is None")
        elif permission is None and folder_to_mount is not None:
            raise ValueError("folder_to_mount provided but permission is None")
        if permission is not None and permission not in ["READ_ONLY", "READ_WRITE"]:
            raise ValueError(
                "permission must be 'READ_ONLY' or 'READ_WRITE' when provided"
            )

        self.image = image
        self.folder_to_mount = folder_to_mount
        self.permission = permission
        self.container = None
        self.client = docker.from_env()
        self.port = port  # required host port
        self.container_port = 8080
        self._create_working_dir()
        self.start()

    def _create_working_dir(self):
        if not os.path.exists(WORKING_DIR):
            os.makedirs(WORKING_DIR)
            logger.info("🗂️ Created working directory at %s", WORKING_DIR)
        else:
            logger.info("🗂️ Working directory already exists at %s", WORKING_DIR)

    def start(self):
        mode_map = {"READ_ONLY": "ro", "READ_WRITE": "rw"}
        volumes_config = {WORKING_DIR: {"bind": DOCKER_WORKING_DIR, "mode": "rw"}}
        if self.folder_to_mount and self.permission:
            if self.permission == "READ_ONLY":
                volumes_config[self.folder_to_mount] = {
                    "bind": f"/ro/{os.path.basename(self.folder_to_mount)}",
                    "mode": mode_map[self.permission],
                }
                logger.info(
                    "📦 Volume mapping: %s → /ro/%s",
                    self.folder_to_mount,
                    os.path.basename(self.folder_to_mount),
                )
            else:
                volumes_config[self.folder_to_mount] = {
                    "bind": f"/{DOCKER_WORKING_DIR}/{os.path.basename(self.folder_to_mount)}",
                    "mode": mode_map[self.permission],
                }
                logger.debug(
                    "📦 Volume mapping: %s → /{DOCKER_WORKING_DIR}/%s",
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
            privileged=True,  # Required for mounting overlayfs
            environment={"BOT_PORT": str(self.container_port)},
        )
        logger.info(
            "🚀 Started container %s with image %s on host port %s",
            self.container.id[:12],
            self.image,
            self.port,
        )
        time.sleep(2)  # Give some time for the server to start

        if self.permission == "READ_ONLY":
            self._setup_overlay_mount(self.folder_to_mount)

    def _setup_overlay_mount(self, folder_to_mount: str):
        path_name = os.path.basename(os.path.abspath(folder_to_mount))
        # Mount /ro/path_name to /{WORKING_DIR}/path_name using overlayfs
        mount_command = (
            f"mkdir -p /overlaydir && "
            f"mkdir -p /{DOCKER_WORKING_DIR}/{path_name} /{DOCKER_WORKING_DIR}/overlay/{path_name}/upper /{DOCKER_WORKING_DIR}/overlay/{path_name}/work && "
            f"mount -t overlay overlay -o lowerdir=/ro/{path_name},upperdir=/{DOCKER_WORKING_DIR}/overlay/{path_name}/upper,workdir=/{DOCKER_WORKING_DIR}/overlay/{path_name}/work /{DOCKER_WORKING_DIR}/{path_name}"
        )
        self.execute(mount_command)
        logger.info(
            "🔒 Set up overlay mount for read-only directory at /{DOCKER_WORKING_DIR}/%s",
            path_name,
        )

    def stop(self):
        """Stop and remove the container"""
        if self.container:
            self.container.stop()
            self.container.remove()
            self.container = None

        # Remove working directory
        if os.path.exists(WORKING_DIR):
            try:
                import shutil

                shutil.rmtree(WORKING_DIR)
                logger.info("🗑️ Removed working directory at %s", WORKING_DIR)
            except Exception as e:
                logger.error("❌ Failed to remove working directory: %s", e)

    def execute(self, command: str, timeout: Optional[int] = 10) -> CmdReturn: # TODO: Need proper return value
        logger.debug("➡️  Executing command in container: %s", command)
        try:
            response = requests.post(
                f"http://localhost:{self.port}/",
                json={"message": command},
                timeout=timeout,
            )
            response.raise_for_status()
            logger.debug("⬅️  Command output: %s", response.json().get("output", ""))
            output = response.json().get("output", "")
            return CmdReturn(
                stdout=output, stderr="", return_code=0
            )
            self.container.reload()
            logger.info("ℹ️ Container status: %s", self.container.status)
            if self.container.status != "running":
                logs = self.container.logs().decode("utf-8", errors="replace")
                logger.error("🛑 Container not running. Recent logs below:\n%s", logs)
            return CmdReturn(stdout="", stderr="Connection error", return_code=1)
        except requests.exceptions.RequestException as e:
            logger.exception("❌ Request failed while executing command: %s", e)
            return CmdReturn(stdout="", stderr=str(e), return_code=1)
        except Exception as e:
            logger.exception("❌ Unexpected error while executing command: %s", e)
            return CmdReturn(stdout="", stderr="Unexpected error", return_code=1)
