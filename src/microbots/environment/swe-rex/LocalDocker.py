import asyncio
import logging
import os
from enum import Enum
from typing import Optional, Final

from Environment.Environment import Environment, CmdReturn
from swerex.deployment.docker import DockerDeployment
from swerex.runtime.abstract import (
    CreateBashSessionRequest,
    CloseBashSessionRequest,
    BashAction,
    Observation,
)

PYTHON_IMAGE = "mcr.microsoft.com/devcontainers/python:3.11"

logger = logging.getLogger(__name__)


class Permission(str, Enum):
    READ_ONLY = "READ_ONLY"
    READ_WRITE = "READ_WRITE"


class LocalDocker(Environment):
    BASE_PATH: Final[str] = "/workdir"

    def _validate_permission_args(
        self,
        folder_to_mount: Optional[str],
        permission: Optional[Permission],
    ):
        if folder_to_mount is None and permission is not None:
            raise ValueError("permission provided but folder_to_mount is None")
        if permission is None and folder_to_mount is not None:
            raise ValueError("folder_to_mount provided but permission is None")
        if permission is not None and permission not in (
            Permission.READ_ONLY,
            Permission.READ_WRITE,
        ):
            raise ValueError(
                "permission must be Permission.READ_ONLY or Permission.READ_WRITE when provided"
            )

    def _get_mount_args(
        self,
        folder_to_mount: Optional[str],
        permission: Optional[Permission],
    ) -> str:
        self._validate_permission_args(folder_to_mount, permission)

        mount_args = ""
        if folder_to_mount and permission:
            sanitized = os.path.abspath(folder_to_mount).strip()
            # For read-only, mount to /ro first, overlay will be set up later
            if permission == Permission.READ_ONLY:
                target_path = f"/ro/{os.path.basename(sanitized)}"
                mode = "ro"
            else:
                target_path = f"{LocalDocker.BASE_PATH}/{os.path.basename(sanitized)}"
                mode = "rw"
            mount_args = f"-v {sanitized}:{target_path}:{mode}"
            logger.info("🥪 Volume mapping: %s -> %s (%s)", sanitized, target_path, mode)
            logger.debug("🗻 Mount args: %r", mount_args)
        return mount_args

    def _get_docker_args(self, mount_args: str = "") -> list[str]:
        # _get_mount_args returns either "" or a single string beginning with -v; split into tokens for DockerDeployment
        docker_args = []

        if mount_args:
            if mount_args.startswith('-v '):
                # split once: '-v src:dest:mode'
                flag, rest = mount_args.split(' ', 1)
                docker_args.extend([flag, rest])
            else:
                docker_args.append(mount_args)

        return docker_args

    def __init__(
        self,
        folder_to_mount: Optional[str] = None,
        permission: Optional[Permission] = Permission.READ_WRITE,
        image: str = PYTHON_IMAGE,
    ):
        self.folder_to_mount = folder_to_mount
        self.permission = permission
        mount_args = self._get_mount_args(folder_to_mount, permission)
        docker_args = self._get_docker_args(mount_args)
        self.deployment = DockerDeployment(image=image, docker_args=docker_args)
        asyncio.run(self.deployment.start())
        self.start()

        # Set up overlay mount for read-only if needed
        if folder_to_mount and permission == Permission.READ_ONLY:
            self._setup_overlay_mount()

        logger.info("🚀 LocalDocker environment initialized successfully")

    def start(self):  # type: ignore[override]
        # Acquire runtime and open a bash session.
        self.runtime = self.deployment.runtime
        asyncio.run(self.runtime.create_session(CreateBashSessionRequest()))

    def _setup_overlay_mount(self):
        """Set up overlay filesystem for read-only mount to allow LLM to create intermediate files."""
        if not self.folder_to_mount:
            return

        path_name = os.path.basename(os.path.abspath(self.folder_to_mount))
        # Mount /ro/path_name to /workdir/path_name using overlayfs
        mount_command = (
            f"mkdir -p /overlaydir && "
            f"mkdir -p {LocalDocker.BASE_PATH}/{path_name} {LocalDocker.BASE_PATH}/overlay/{path_name}/upper {LocalDocker.BASE_PATH}/overlay/{path_name}/work && "
            f"mount -t overlay overlay -o lowerdir=/ro/{path_name},upperdir={LocalDocker.BASE_PATH}/overlay/{path_name}/upper,workdir={LocalDocker.BASE_PATH}/overlay/{path_name}/work {LocalDocker.BASE_PATH}/{path_name}"
        )
        result = asyncio.run(self.execute(mount_command))
        if result.return_code != 0:
            logger.warning("Failed to set up overlay mount: %s", result.stderr)
        else:
            logger.info(
                "🔒 Set up overlay mount for read-only directory at %s/%s",
                LocalDocker.BASE_PATH,
                path_name,
            )

    def stop(self):
        try:
            asyncio.run(self.runtime.close_session(CloseBashSessionRequest()))
        finally:
            asyncio.run(self.deployment.stop())

    async def execute(
        self, command: str, timeout: Optional[int] = 300
    ) -> CmdReturn:
        """Execute a shell command inside the container.

        We pass the command through bash -lc to support shell features (globbing, env vars, pipelines).
        """
        logger.debug("🔧 Executing command: %s", command)
        try:
            output: Observation = await asyncio.wait_for(
                self.runtime.run_in_session(BashAction(command=command)), timeout
            )
            logger.debug("📋 Command '%s' completed:", command)
            logger.debug("   ├─ 📤 Exit code: %s", output.exit_code)
            logger.debug("   ├─ 📝 Output: %s", output.output[:100] + "..." if len(output.output) > 100 else output.output)
            logger.debug("   └─ ⚠️ Error: %s", output.failure_reason if output.failure_reason else "(none)")
            return CmdReturn(
                stdout=output.output,
                return_code=output.exit_code,
                stderr=output.failure_reason if output.failure_reason else "",
            )
        except asyncio.TimeoutError:
            logger.error("⏱️ Command timed out after %s seconds: '%s'", timeout, command)
            # Try to kill the timed-out process to maintain session usability
            try:
                # Send SIGINT (Ctrl+C) to interrupt the running command
                logger.debug("Attempting to interrupt timed-out command")
                await self.runtime.run_in_session(BashAction(command="\x03"))  # Ctrl+C
            except Exception as e:
                logger.debug("Could not interrupt timed-out command: %s", e)

            return CmdReturn(
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                return_code=124,  # Standard timeout exit code
            )
        except Exception as e:
            logger.error("❌ Error occurred while executing command '%s': %s", command, e)
            return CmdReturn(
            stdout="",
            stderr=str(e),
            return_code=1,
            )


