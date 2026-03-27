"""
CopilotBot — A wrapper around the GitHub Copilot SDK.

Unlike MicroBot (which manages its own LLM ↔ shell agent loop), CopilotBot
delegates the **entire agent loop to the Copilot runtime**.  Copilot handles
planning, tool invocation (file edits, shell commands, web requests, etc.)
and multi-turn reasoning autonomously.

CopilotBot reuses the Microbots infrastructure:
  - Docker sandbox (LocalDockerEnvironment) for isolated execution
  - Mount system for folder access control
  - ToolAbstract lifecycle (install → verify → setup) for additional tools
  - copilot-cli is installed **inside** the container and run in headless
    server mode; the SDK on the host connects to it via TCP.

Architecture:

    Host                          Docker Container
    ─────                         ────────────────
    CopilotBot                    copilot-cli --headless --port <P>
        │                              │
        ├── Copilot SDK ──TCP──────────┘
        │   (ExternalServerConfig)
        │
        ├── additional tools
        │   (define_tool → SDK session)
        │
        └── BotRunResult

Prerequisites:
  - pip install microbots[ghcp]   (github-copilot-sdk)
  - Docker daemon running
  - GitHub authentication (GITHUB_TOKEN / COPILOT_GITHUB_TOKEN or copilot login)
"""

import asyncio
import os
import time
import threading
from logging import getLogger
from typing import Optional

from microbots.constants import (
    DOCKER_WORKING_DIR,
    PermissionLabels,
)
from microbots.environment.local_docker.LocalDockerEnvironment import (
    LocalDockerEnvironment,
)
from microbots.extras.mount import Mount, MountType
from microbots.MicroBot import BotRunResult
from microbots.tools.tool import ToolAbstract
from microbots.utils.network import get_free_port

logger = getLogger(" CopilotBot ")

# Default model when none is specified (just the deployment name, no provider prefix)
_DEFAULT_MODEL = "gpt-4.1"

# Time (seconds) to wait for copilot-cli to start inside the container
_CLI_STARTUP_TIMEOUT = 60

# copilot-cli port inside the container
_CONTAINER_CLI_PORT = 4321


class CopilotBot:
    """Wrapper around the GitHub Copilot SDK with a sandboxed Docker environment.

    The Copilot runtime manages the agent loop (planning, tool calls,
    multi-turn reasoning).  CopilotBot sets up the sandbox, installs
    copilot-cli inside it, connects the SDK, and exposes a simple
    ``run(task)`` interface.

    Parameters
    ----------
    model : str
        Copilot model name (e.g. ``"gpt-4.1"``, ``"claude-sonnet-4.5"``).
        Unlike MicroBot, no ``<provider>/`` prefix is needed.
    folder_to_mount : str
        Absolute host path to mount into the sandbox.
    permission : PermissionLabels
        Mount permission — READ_ONLY or READ_WRITE.  Defaults to READ_WRITE.
    environment : Optional[LocalDockerEnvironment]
        Pre-created environment.  One is created automatically when *None*.
    additional_tools : Optional[list[ToolAbstract]]
        Extra Microbots tools to install in the sandbox.  Their
        ``usage_instructions_to_llm`` are appended to the system message
        and, where possible, they are registered as SDK custom tools.
    github_token : Optional[str]
        Explicit GitHub token.  Falls back to ``GITHUB_TOKEN`` /
        ``COPILOT_GITHUB_TOKEN`` env vars.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        folder_to_mount: Optional[str] = None,
        permission: PermissionLabels = PermissionLabels.READ_WRITE,
        environment: Optional[LocalDockerEnvironment] = None,
        additional_tools: Optional[list[ToolAbstract]] = None,
        github_token: Optional[str] = None,
    ):
        try:
            from copilot import CopilotClient, ExternalServerConfig
            from copilot.session import PermissionHandler
        except ImportError:
            raise ImportError(
                "CopilotBot requires the github-copilot-sdk package. "
                "Install with: pip install microbots[ghcp]"
            )

        self.model = model
        self.additional_tools = additional_tools or []
        self.github_token = (
            github_token
            or os.environ.get("COPILOT_GITHUB_TOKEN")
            or os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GH_TOKEN")
        )

        # ── Mount setup ─────────────────────────────────────────────
        self.folder_to_mount: Optional[Mount] = None
        if folder_to_mount:
            sandbox_path = f"/{DOCKER_WORKING_DIR}/{os.path.basename(folder_to_mount)}"
            self.folder_to_mount = Mount(folder_to_mount, sandbox_path, permission)

        # ── Docker environment ──────────────────────────────────────
        self.environment = environment
        if not self.environment:
            self._create_environment()

        # ── Install additional tools inside the container ───────────
        for tool in self.additional_tools:
            tool.install_tool(self.environment)
            tool.verify_tool_installation(self.environment)

        # ── Install & start copilot-cli inside the container ────────
        self._cli_host_port = get_free_port()
        self._install_copilot_cli()
        self._start_copilot_cli_server()

        # ── Background event loop for async SDK calls ───────────────
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

        # ── Connect SDK to in-container CLI ─────────────────────────
        self._client = CopilotClient(
            ExternalServerConfig(url=f"localhost:{self._cli_host_port}")
        )
        self._run_async(self._client.start())
        self._PermissionHandler = PermissionHandler

        logger.info(
            "✅ CopilotBot initialised — model=%s, cli_port=%d",
            self.model,
            self._cli_host_port,
        )

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def run(
        self,
        task: str,
        additional_mounts: Optional[list[Mount]] = None,
        timeout_in_seconds: int = 600,
        streaming: bool = False,
    ) -> BotRunResult:
        """Send *task* to the Copilot agent and wait for completion.

        The Copilot runtime manages the full agent loop autonomously —
        planning, tool invocation, multi-turn reasoning, and file edits
        all happen inside the sandboxed environment.

        Parameters
        ----------
        task : str
            A natural-language description of the task.
        additional_mounts : Optional[list[Mount]]
            Extra folders to copy into the container before running.
        timeout_in_seconds : int
            Maximum wall-clock time for the agent run.
        streaming : bool
            Whether to enable streaming delta events (logged at DEBUG level).

        Returns
        -------
        BotRunResult
            status=True on success with the agent's final message in *result*,
            or status=False with an error description.
        """
        # Setup additional tools (env vars, files, setup_commands)
        for tool in self.additional_tools:
            tool.setup_tool(self.environment)

        # Mount additional folders
        for mount in additional_mounts or []:
            self._mount_additional(mount)

        # Build system message with tool instructions
        system_content = self._build_system_message()

        # Build SDK custom tools from additional_tools
        sdk_tools = self._build_sdk_tools()

        try:
            result_text = self._run_async(
                self._execute_session(
                    task=task,
                    system_content=system_content,
                    sdk_tools=sdk_tools,
                    timeout=timeout_in_seconds,
                    streaming=streaming,
                )
            )
            return BotRunResult(status=True, result=result_text, error=None)
        except Exception as e:
            logger.exception("❌ CopilotBot run failed: %s", e)
            return BotRunResult(status=False, result=None, error=str(e))

    def stop(self):
        """Tear down the SDK client, CLI server, and Docker environment."""
        if getattr(self, "_stopped", False):
            return
        self._stopped = True

        # Stop the SDK client (best-effort, with timeout to avoid deadlock)
        try:
            if self._loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self._client.stop(), self._loop
                )
                future.result(timeout=10)
        except Exception:
            pass

        # Shut down the background event loop
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)
        except Exception:
            pass

        if self.environment:
            self.environment.stop()
            self.environment = None
        logger.info("🛑 CopilotBot stopped")

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────
    # Private — environment & CLI setup
    # ──────────────────────────────────────────────────────────────────

    def _create_environment(self):
        free_port = get_free_port()
        # Also map the copilot-cli headless port
        self._cli_host_port = get_free_port()
        self.environment = LocalDockerEnvironment(
            port=free_port,
            folder_to_mount=self.folder_to_mount,
        )
        # Expose additional port mapping for copilot-cli
        self._map_cli_port()

    def _map_cli_port(self):
        """Add a second port mapping for the copilot-cli headless server.

        Docker port mappings are static after container creation, so we use
        ``socat`` inside the container to forward the CLI port through the
        existing shell_server port range, OR we use ``docker exec`` via iptables.

        The simplest reliable approach: install socat and forward from a known
        port that's already exposed, or use ``docker port``.

        Actually, the cleanest approach: stop the container, recreate it with
        the additional port.  Since we control environment creation this is safe.
        """
        # The environment was just created by us, so recreating with an extra port
        # is acceptable.  We stop the existing container and create a new one
        # with both ports mapped.
        if not self.environment.container:
            return

        container = self.environment.container
        image = self.environment.image
        port = self.environment.port
        container_port = self.environment.container_port

        # Gather existing volume config from the running container
        import docker

        container.stop()
        container.remove()

        # Re-create with both ports
        volumes_config = {self.environment.working_dir: {"bind": DOCKER_WORKING_DIR, "mode": "rw"}}
        if self.folder_to_mount:
            mode_map = {"READ_ONLY": "ro", "READ_WRITE": "rw"}
            if self.folder_to_mount.permission == PermissionLabels.READ_ONLY:
                volumes_config[self.folder_to_mount.host_path_info.abs_path] = {
                    "bind": f"/ro/{os.path.basename(self.folder_to_mount.sandbox_path)}",
                    "mode": mode_map[self.folder_to_mount.permission],
                }
            else:
                volumes_config[self.folder_to_mount.host_path_info.abs_path] = {
                    "bind": self.folder_to_mount.sandbox_path,
                    "mode": mode_map[self.folder_to_mount.permission],
                }

        port_mapping = {
            f"{container_port}/tcp": port,
            f"{_CONTAINER_CLI_PORT}/tcp": self._cli_host_port,
        }

        client = docker.from_env()
        self.environment.container = client.containers.run(
            image,
            volumes=volumes_config,
            ports=port_mapping,
            detach=True,
            working_dir="/app",
            privileged=True,
            environment={"BOT_PORT": str(container_port)},
        )
        logger.info(
            "🚀 Recreated container with CLI port mapping: host %d → container %d",
            self._cli_host_port,
            _CONTAINER_CLI_PORT,
        )
        time.sleep(2)

        # Re-setup overlay if needed
        if self.folder_to_mount and self.folder_to_mount.permission == PermissionLabels.READ_ONLY:
            self.environment._setup_overlay_mount()

        # cd into mounted folder
        if self.folder_to_mount:
            self.environment.execute(f"cd {self.folder_to_mount.sandbox_path}")
        else:
            self.environment.execute("cd /")

    def _install_copilot_cli(self):
        """Install copilot-cli inside the Docker container."""
        logger.info("📦 Installing copilot-cli inside container...")

        # Install Node.js (required for copilot-cli via npm)
        install_commands = [
            # Install Node.js 22.x (copilot-cli requires Node 22+)
            "apt-get update -qq && apt-get install -y -qq curl ca-certificates > /dev/null 2>&1",
            "curl -fsSL https://deb.nodesource.com/setup_22.x | bash - > /dev/null 2>&1",
            "apt-get install -y -qq nodejs > /dev/null 2>&1",
            # Install copilot-cli globally
            "npm install -g @github/copilot > /dev/null 2>&1",
        ]

        for cmd in install_commands:
            result = self.environment.execute(cmd, timeout=300)
            if result.return_code != 0:
                raise RuntimeError(
                    f"Failed to install copilot-cli: {cmd}\n"
                    f"stdout: {result.stdout}\nstderr: {result.stderr}"
                )

        # Verify installation
        verify = self.environment.execute("copilot --version")
        if verify.return_code != 0:
            raise RuntimeError(
                f"copilot-cli installation verification failed: {verify.stderr}"
            )
        logger.info("✅ copilot-cli installed: %s", verify.stdout.strip())

    def _start_copilot_cli_server(self):
        """Start copilot-cli in headless server mode inside the container.

        The CLI listens on ``_CONTAINER_CLI_PORT`` which is mapped to
        ``self._cli_host_port`` on the host.  Authentication is handled
        via the GITHUB_TOKEN environment variable injected into the container.
        """
        # Inject the GitHub token into the container for authentication
        if self.github_token:
            self.environment.execute(
                f'export GITHUB_TOKEN="{self.github_token}"'
            )
            self.environment.execute(
                f'export COPILOT_GITHUB_TOKEN="{self.github_token}"'
            )

        # Start copilot in headless mode in the background
        # Using nohup + & to run it as a background process inside the container's shell
        start_cmd = (
            f"nohup copilot --headless --port {_CONTAINER_CLI_PORT} "
            f"> /var/log/copilot-cli.log 2>&1 &"
        )
        result = self.environment.execute(start_cmd)
        if result.return_code != 0:
            raise RuntimeError(
                f"Failed to start copilot-cli server: {result.stderr}"
            )

        # Wait for the server to be ready
        self._wait_for_cli_ready()
        logger.info(
            "✅ copilot-cli headless server running on container port %d (host port %d)",
            _CONTAINER_CLI_PORT,
            self._cli_host_port,
        )

    def _wait_for_cli_ready(self):
        """Poll until the copilot-cli server is accepting connections."""
        import socket as _socket

        deadline = time.time() + _CLI_STARTUP_TIMEOUT
        while time.time() < deadline:
            try:
                sock = _socket.create_connection(
                    ("localhost", self._cli_host_port), timeout=2
                )
                sock.close()
                return
            except (ConnectionRefusedError, OSError):
                time.sleep(1)
        raise TimeoutError(
            f"copilot-cli did not become ready within {_CLI_STARTUP_TIMEOUT}s "
            f"on host port {self._cli_host_port}"
        )

    # ──────────────────────────────────────────────────────────────────
    # Private — SDK session & execution
    # ──────────────────────────────────────────────────────────────────

    def _run_async(self, coro):
        """Submit an async coroutine to the background loop and block."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def _execute_session(
        self,
        task: str,
        system_content: str,
        sdk_tools: list,
        timeout: int,
        streaming: bool,
    ) -> str:
        """Create a Copilot session, send the task, and collect the result."""
        from copilot.generated.session_events import SessionEventType

        session_kwargs = {
            "model": self.model,
            "on_permission_request": self._PermissionHandler.approve_all,
            "streaming": streaming,
        }

        if system_content:
            session_kwargs["system_message"] = {"content": system_content}

        if sdk_tools:
            session_kwargs["tools"] = sdk_tools

        session = await self._client.create_session(**session_kwargs)

        collected_text = []
        done_event = asyncio.Event()

        def _on_event(event):
            if event.type == SessionEventType.ASSISTANT_MESSAGE:
                if event.data and event.data.content:
                    collected_text.append(event.data.content)
            elif event.type == SessionEventType.ASSISTANT_MESSAGE_DELTA:
                if event.data and event.data.delta_content:
                    logger.debug("📝 %s", event.data.delta_content)
            elif event.type == SessionEventType.SESSION_IDLE:
                done_event.set()

        session.on(_on_event)

        # Send the task prompt and wait for completion
        response = await session.send_and_wait(task, timeout=float(timeout))

        # If send_and_wait returned a full response, use it
        if response and response.data and response.data.content:
            return response.data.content

        # Otherwise wait for the collected events
        if not collected_text:
            try:
                await asyncio.wait_for(done_event.wait(), timeout=float(timeout))
            except asyncio.TimeoutError:
                pass

        await session.disconnect()

        if collected_text:
            return collected_text[-1]  # Return the last assistant message

        return "Agent completed without producing a final message."

    def _build_system_message(self) -> str:
        """Compose the system message from mount info and tool instructions."""
        parts = []

        if self.folder_to_mount:
            parts.append(
                f"The working directory is mounted at {self.folder_to_mount.sandbox_path}. "
                f"You can access files using paths relative to or absolute from that directory."
            )

        for tool in self.additional_tools:
            if tool.usage_instructions_to_llm:
                parts.append(tool.usage_instructions_to_llm)

        return "\n\n".join(parts)

    def _build_sdk_tools(self) -> list:
        """Convert Microbots additional tools into Copilot SDK tool definitions.

        Only tools that implement ``is_invoked`` / have an ``invoke`` method
        (ExternalTools) can be meaningfully wrapped.  Internal tools that run
        via shell commands are already accessible to Copilot's built-in shell
        tool and don't need explicit registration.
        """
        from microbots.tools.external_tool import ExternalTool

        sdk_tools = []
        for tool in self.additional_tools:
            if isinstance(tool, ExternalTool) and hasattr(tool, "invoke"):
                sdk_tool = self._wrap_external_tool(tool)
                if sdk_tool:
                    sdk_tools.append(sdk_tool)
        return sdk_tools

    def _wrap_external_tool(self, tool: ToolAbstract):
        """Wrap a Microbots ExternalTool as a Copilot SDK define_tool."""
        try:
            from copilot.tools import Tool as CopilotTool, ToolInvocation, ToolResult
        except ImportError:
            return None

        bot_ref = self  # Capture reference for the handler closure

        async def handler(invocation: ToolInvocation) -> ToolResult:
            command = invocation.arguments.get("command", "")
            try:
                cmd_return = tool.invoke(command, bot_ref)
                output = cmd_return.stdout if cmd_return.return_code == 0 else (
                    f"COMMAND FAILED (rc={cmd_return.return_code})\n"
                    f"stdout: {cmd_return.stdout}\nstderr: {cmd_return.stderr}"
                )
                return ToolResult(
                    text_result_for_llm=output,
                    result_type="success" if cmd_return.return_code == 0 else "failure",
                )
            except Exception as e:
                return ToolResult(
                    text_result_for_llm=f"Tool error: {e}",
                    result_type="failure",
                )

        return CopilotTool(
            name=tool.name,
            description=tool.description,
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": f"The command to invoke the {tool.name} tool",
                    },
                },
                "required": ["command"],
            },
            handler=handler,
        )

    # ──────────────────────────────────────────────────────────────────
    # Private — mount helpers
    # ──────────────────────────────────────────────────────────────────

    def _mount_additional(self, mount: Mount):
        """Copy an additional folder into the running container."""
        if mount.mount_type != MountType.COPY:
            raise ValueError(
                "Only COPY mount type is supported for additional mounts"
            )
        if not self.environment.copy_to_container(
            mount.host_path_info.abs_path, mount.sandbox_path
        ):
            raise ValueError(
                f"Failed to copy additional mount: "
                f"{mount.host_path_info.abs_path} -> {mount.sandbox_path}"
            )
