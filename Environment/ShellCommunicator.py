#!/usr/bin/env python3
"""
Shell Communication Script
A Python script to create and communicate with shell sessions.
Supports interactive shell communication, command execution, and bidirectional data flow.
"""

import os
import queue
import subprocess
import sys
import threading
import time
import logging
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class ShellCommunicator:
    """
    A class to create and manage shell sessions with bidirectional communication.
    """

    def __init__(self, shell_type: str = "bash", encoding: str = "utf-8"):
        """
        Initialize the shell communicator.

        Args:
            shell_type: Type of shell ("powershell", "cmd", "bash", "python")
            encoding: Text encoding for communication
        """
        self.shell_type = shell_type.lower()
        self.encoding = encoding
        self.process: Optional[subprocess.Popen] = None
        self.output_queue = queue.Queue()
        self.error_queue = queue.Queue()
        self.is_running = False
        self.output_thread: Optional[threading.Thread] = None
        self.error_thread: Optional[threading.Thread] = None
        self.output_callback: Optional[Callable] = None

        # Define shell commands
        self.shell_commands = {
            "powershell": ["powershell.exe", "-NoLogo", "-NoExit"],
            "cmd": ["cmd.exe", "/k"],
            "bash": ["bash"],
            "python": [sys.executable, "-i"],
            "wsl": ["wsl.exe"],
        }

    def start_session(self) -> bool:
        """
        Start a new shell session.

        Returns:
            bool: True if session started successfully, False otherwise
        """
        try:
            if self.shell_type not in self.shell_commands:
                logger.error("🛑 Unsupported shell type: %s", self.shell_type)
                return False

            cmd = self.shell_commands[self.shell_type]

            # Create the subprocess
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding=self.encoding,
                bufsize=0,
                universal_newlines=True,
            )

            self.is_running = True

            # Start output monitoring threads
            self.output_thread = threading.Thread(
                target=self._monitor_output,
                args=(self.process.stdout, self.output_queue, "OUTPUT"),
                daemon=True,
            )
            self.error_thread = threading.Thread(
                target=self._monitor_output,
                args=(self.process.stderr, self.error_queue, "ERROR"),
                daemon=True,
            )

            self.output_thread.start()
            self.error_thread.start()

            logger.info("🚀 %s session started successfully", self.shell_type.capitalize())
            logger.debug("🆔 Process ID: %s", self.process.pid)
            return True

        except Exception as e:
            logger.exception("❌ Failed to start shell session: %s", e)
            return False

    def _monitor_output(self, stream, output_queue: queue.Queue, stream_type: str):
        """
        Monitor shell output in a separate thread.

        Args:
            stream: The stream to monitor (stdout or stderr)
            output_queue: Queue to store output
            stream_type: Type of stream ("OUTPUT" or "ERROR")
        """
        try:
            while self.is_running and self.process and self.process.poll() is None:
                line = stream.readline()
                if line:
                    output_queue.put((stream_type, line.rstrip()))
                    if self.output_callback:
                        self.output_callback(stream_type, line.rstrip())
                elif self.process.poll() is not None:
                    break
        except Exception as e:
            output_queue.put((stream_type, f"Monitor error: {e}"))

    def send_command(
        self, command: str, wait_for_output: bool = True, timeout: float = 5.0
    ) -> List[str]:
        """
        Send a command to the shell session.

        Args:
            command: Command to execute
            wait_for_output: Whether to wait for command output
            timeout: Timeout for waiting for output

        Returns:
            List of output lines
        """
        if not self.is_running or not self.process:
            logger.warning("⚠️ No active shell session")
            return []

        try:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
            logger.debug("➡️ Sent command: %s", command)

            if not wait_for_output:
                return []

            output_lines = []
            start_time = time.time()

            while time.time() - start_time < timeout:
                try:
                    # Check for output
                    stream_type, line = self.output_queue.get(timeout=0.1)
                    output_lines.append(f"{line}")
                    if stream_type == "ERROR":
                        logger.error("❌ %s", line)
                    else:
                        logger.debug("📤 %s", line)
                except queue.Empty:
                    continue
                except Exception:
                    logger.exception("❌ Unexpected error while reading output queue")
                    break

            # Check for errors
            try:
                while True:
                    stream_type, line = self.error_queue.get_nowait()
                    output_lines.append(f"{line}")
                    if stream_type == "ERROR":
                        logger.error("❌ %s", line)
                    else:
                        logger.debug("📤 %s", line)
            except queue.Empty:
                pass

            return output_lines

        except Exception as e:
            logger.exception("❌ Failed to send command: %s", e)
            return []


    def is_alive(self) -> bool:
        """
        Check if the shell session is still alive.

        Returns:
            bool: True if session is active, False otherwise
        """
        return (
            self.is_running and self.process is not None and self.process.poll() is None
        )

    def get_shell_info(self) -> dict:
        """
        Get information about the current shell session.

        Returns:
            Dictionary with shell session information
        """
        if not self.process:
            return {"status": "Not started"}

        return {
            "shell_type": self.shell_type,
            "pid": self.process.pid,
            "status": "Running" if self.is_alive() else "Stopped",
            "encoding": self.encoding,
            "return_code": self.process.returncode,
        }

    def close_session(self):
        """
        Close the shell session and cleanup resources.
        """
        logger.info("🛑 Closing shell session…")

        self.is_running = False

        if self.process:
            try:
                # Try to terminate gracefully
                if self.shell_type == "powershell":
                    self.send_command("exit", wait_for_output=False)
                elif self.shell_type == "cmd":
                    self.send_command("exit", wait_for_output=False)
                else:
                    self.send_command("exit", wait_for_output=False)

                # Wait a bit for graceful shutdown
                time.sleep(1)

                # Force terminate if still running
                if self.process.poll() is None:
                    self.process.terminate()
                    time.sleep(1)

                    if self.process.poll() is None:
                        self.process.kill()

                logger.info("✅ Shell session closed")

            except Exception as e:
                logger.exception("⚠️ Error during cleanup: %s", e)

        # Wait for threads to finish
        if self.output_thread and self.output_thread.is_alive():
            self.output_thread.join(timeout=2)
        if self.error_thread and self.error_thread.is_alive():
            self.error_thread.join(timeout=2)
