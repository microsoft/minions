import logging
import os
from typing import Optional

from microbots.constants import DOCKER_WORKING_DIR, PermissionLabels
from microbots.MicroBot import BotType, MicroBot, system_prompt_common
from microbots.tools.tool import ToolAbstract
from microbots.tools.tool_yaml_parser import parse_tool_definition
from microbots.extras.mount import Mount

logger = logging.getLogger(__name__)

# The sub_agent YAML ships with the package — resolve it once at import time.
_SUB_AGENT_YAML = "sub_agent.yaml"


class LeadBot(MicroBot):
    """A leadership bot that decomposes a complex task into subtasks
    and delegates each one to a ``sub_agent`` running inside the sandbox.

    Workflow
    --------
    1. The user provides a high-level task.
    2. LeadBot's system prompt instructs the LLM to break the task into
       ordered subtasks and invoke ``sub_agent`` for each one.
    3. Each ``sub_agent`` call spawns an autonomous ReadingBot or WritingBot
       inside the same sandbox to solve a single subtask.
    4. After all subtasks are completed, the LLM synthesises the results
       into a final answer.
    """

    def __init__(
        self,
        model: str,
        folder_to_mount: str,
        environment: Optional[any] = None,
        additional_tools: Optional[list[ToolAbstract]] = None,
    ):
        """
        Parameters
        ----------
        model : str
            Model identifier in ``<provider>/<deployment>`` format.
        folder_to_mount : str
            Absolute host path to the repository the bot should operate on.
        environment : Optional[any]
            Pre-created execution environment.  A ``LocalDockerEnvironment``
            is created automatically when *None*.
        additional_tools : Optional[list[ToolAbstract]]
            Extra tools to install alongside the built-in ``sub_agent`` tool.
        """
        if additional_tools is None:
            additional_tools = []

        # Always include the sub_agent external tool.
        sub_agent_tool = parse_tool_definition(_SUB_AGENT_YAML)
        additional_tools = [sub_agent_tool] + additional_tools

        base_name = os.path.basename(folder_to_mount)
        folder_mount_info = Mount(
            folder_to_mount,
            f"/{DOCKER_WORKING_DIR}/{base_name}",
            PermissionLabels.READ_WRITE,
        )

        system_prompt = f"""
{system_prompt_common}

You are a **Lead Bot** — a senior technical lead responsible for solving a complex task by breaking it down into smaller, focused subtasks and delegating each one to a sub-agent.

## Your environment
- The repository is mounted at `{folder_mount_info.sandbox_path}`.
- You have access to a `sub_agent` CLI tool inside this environment.

## sub_agent usage
```
sub_agent --repo_path {folder_mount_info.sandbox_path} --permission <readonly|write> --task "<task description>"
```

- Use `--permission readonly` for investigation / analysis tasks (e.g. finding root causes, reading code).
- Use `--permission write` for modification tasks (e.g. fixing bugs, refactoring code).
- Always provide a clear, self-contained `--task` description so that the sub-agent can work autonomously without extra context.

## Your workflow
1. **Analyse** the task carefully. Identify what information you need and what changes are required.
2. **Decompose** the task into a numbered plan of subtasks (write them in your `thoughts`).
3. **Execute** each subtask one at a time by invoking `sub_agent` with the appropriate permission and a detailed task description.
4. After each sub_agent call, **review** its output before moving on. If a subtask failed, analyse the error and retry with a corrected task description.
5. When **all subtasks are complete**, set `task_done` to true and provide a comprehensive summary of everything that was done and the final outcome in `thoughts`.

## Important rules
- Run only ONE `sub_agent` command at a time. Wait for the output before issuing the next one.
- Never perform the actual work yourself (e.g. editing files directly). Always delegate to `sub_agent`.
- Keep subtasks focused and small — each sub_agent should do one thing well.
- Always pass the same repository path to all sub_agent calls. This ensures they are all working in the same environment and can build on each other's results.
"""

        super().__init__(
            model=model,
            bot_type=BotType.CUSTOM_BOT,
            system_prompt=system_prompt,
            environment=environment,
            additional_tools=additional_tools,
            folder_to_mount=folder_mount_info,
        )

    def run(
        self,
        task: str,
        max_iterations: int = 50,
        timeout_in_seconds: int = 1200,
    ) -> any:
        """Solve *task* by decomposing it into subtasks and delegating to sub-agents.

        Parameters
        ----------
        task : str
            High-level task description.
        max_iterations : int
            Maximum LLM ↔ shell round-trips (default 50 — higher than a
            regular bot because the lead orchestrates multiple sub-agents).
        timeout_in_seconds : int
            Wall-clock timeout in seconds (default 1200 = 20 min).

        Returns
        -------
        BotRunResult
            The final status, result, and any error message.
        """
        lead_task_prompt = f"""
You are the Lead Bot. Solve the following task by decomposing it into subtasks and delegating each one to a sub_agent.

Task:
{task}
"""
        return super().run(
            task=lead_task_prompt,
            max_iterations=max_iterations,
            timeout_in_seconds=timeout_in_seconds,
        )
