#!/usr/bin/env python3
"""
Sub-agent script that creates a ReadingBot or WritingBot based on
the requested permission level and executes a task autonomously.

Usage:
    sub_agent --repo_path /path/to/repo --permission readonly --task "Analyze the bug"
    sub_agent --repo_path /path/to/repo --permission write --task "Fix the bug"
"""

import argparse
import logging
import os
import sys
import uuid

from pathlib import Path

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Sudo / root privilege check
# ---------------------------------------------------------------------------
def _check_and_elevate():
    """Ensure the process is running with root privileges.

    Writing logs to /var/log/<sub_agent_name>/ requires elevated access.
    If not already root, re-exec the script under sudo. If sudo is
    unavailable or the user cannot authenticate, report the error and exit.
    """
    if os.geteuid() == 0:
        return  # already root

    # Attempt to re-execute under sudo
    sudo_path = "/usr/bin/sudo"
    if not os.path.isfile(sudo_path):
        print(
            "Error: this script requires root privileges to write logs to "
            "/var/log/ but 'sudo' was not found on this system.",
            file=sys.stderr,
        )
        sys.exit(-1)

    try:
        os.execvp("sudo", ["sudo", sys.executable] + sys.argv)
    except OSError as exc:
        print(
            f"Error: failed to elevate privileges via sudo: {exc}",
            file=sys.stderr,
        )
        sys.exit(-1)


_check_and_elevate()

# ---------------------------------------------------------------------------
# Name & log directory setup
# ---------------------------------------------------------------------------
_unique_id = uuid.uuid4().hex[:4].upper()
SUB_AGENT_NAME = f"SubAgent_{_unique_id}"

LOG_DIR = Path(f"/var/log/{SUB_AGENT_NAME}")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging configuration — info + debug logs written under LOG_DIR
# ---------------------------------------------------------------------------
logger = logging.getLogger(SUB_AGENT_NAME)
logger.setLevel(logging.DEBUG)

# File handler — captures DEBUG and above
_file_handler = logging.FileHandler(LOG_DIR / "sub_agent.log")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
)
logger.addHandler(_file_handler)

# Console handler — captures INFO and above
_console_handler = logging.StreamHandler(sys.stderr)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
)
logger.addHandler(_console_handler)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="sub_agent",
        description="Microbots-based sub-agent that performs tasks inside a sandboxed environment.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--repo_path",
        required=True,
        help="Absolute path to the repository to operate on.",
    )
    parser.add_argument(
        "--permission",
        required=True,
        choices=["readonly", "write"],
        help="Permission level: 'readonly' creates a ReadingBot, 'write' creates a WritingBot.",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="The task description for the sub-agent to execute.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------
def main(argv=None):
    args = _parse_args(argv)

    repo_path = args.repo_path
    permission = args.permission
    task = args.task

    logger.info("Starting %s", SUB_AGENT_NAME)
    logger.info("Repo path : %s", repo_path)
    logger.info("Permission: %s", permission)
    logger.info("Task      : %s", task)
    logger.debug("Log directory: %s", LOG_DIR)

    # ---- Validate repo path ------------------------------------------------
    if not os.path.isdir(repo_path):
        logger.error("Repo path does not exist or is not a directory: %s", repo_path)
        print(f"Error: repo_path '{repo_path}' is not a valid directory.", file=sys.stderr)
        sys.exit(-1)

    # ---- Resolve model from environment variable ---------------------------
    model_name = os.environ.get("AZURE_OPENAI_LLM")
    if not model_name:
        logger.error("Environment variable AZURE_OPENAI_LLM is not set.")
        print("Error: environment variable AZURE_OPENAI_LLM is not set.", file=sys.stderr)
        sys.exit(-1)

    model = f"azure-openai/{model_name}"
    logger.info("Using model: %s", model)

    # ---- Create the appropriate bot ----------------------------------------
    try:
        if permission == "readonly":
            from microbots import ReadingBot

            logger.info("Creating ReadingBot for read-only access")
            bot = ReadingBot(model=model, folder_to_mount=repo_path)
        else:
            from microbots import WritingBot

            logger.info("Creating WritingBot for read-write access")
            bot = WritingBot(model=model, folder_to_mount=repo_path)
    except Exception as exc:
        logger.exception("Failed to create bot: %s", exc)
        print(f"Error creating bot: {exc}", file=sys.stderr)
        sys.exit(-1)

    # ---- Run the task (timeout 20 min, max 50 iterations) ------------------
    timeout_seconds = 20 * 60  # 20 minutes
    max_iterations = 50

    logger.info(
        "Running task with timeout=%ds, max_iterations=%d",
        timeout_seconds,
        max_iterations,
    )

    try:
        result = bot.run(
            task=task,
            max_iterations=max_iterations,
            timeout_in_seconds=timeout_seconds,
        )
    except Exception as exc:
        logger.exception("Task execution raised an exception: %s", exc)
        print(f"Error during task execution: {exc}", file=sys.stderr)
        sys.exit(-1)

    # ---- Handle result -----------------------------------------------------
    if result.status:
        logger.info("Task completed successfully.")
        logger.info("Result: %s", result.result)
        print(result.result or "Task completed successfully.")
        sys.exit(0)
    else:
        logger.error("Task failed. Error: %s", result.error)
        print(f"Task failed: {result.error}", file=sys.stderr)
        sys.exit(-1)


if __name__ == "__main__":
    main()
