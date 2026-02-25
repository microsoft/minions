from dataclasses import Field, dataclass
import logging

from microbots.tools.external_tool import ExternalTool
from microbots.MicroBot import system_prompt_common, MicroBot, BotRunResult

logger = logging.getLogger(" 🤖 MicroBot-Sub")

INSTRUCTIONS_TO_LLM = """
Invoke this tool to create a sub-agent that can perform specific tasks as directed by the parent MicroBot. The sub-agent will run in the same sandbox environment and can communicate with the parent MicroBot via standard input/output. Use this tool when you want to delegate a specific task to a sub-agent while maintaining control and oversight over its actions.

Example Usage:
```
    microbot_sub --task "Analyze the contents of the /data directory and report any files larger than 100MB." --iterations 25 --timeout 300
```
In this example, the parent MicroBot would invoke the `microbot_sub` tool to create a sub-agent tasked with analyzing the contents of the `/data` directory and reporting any files larger than 100MB. The sub-agent would have up to 25 iterations and a timeout of 300 seconds to complete the task before it must report back to the parent MicroBot with its findings or any issues encountered.
"""

@dataclass
class MicrobotSubAgent(ExternalTool):
    """
    A MicroBotSubAgent is a specialized ExternalTool that represents a sub-agent running in the same sandbox environment.

    It inherits all behaviors of an ExternalTool, but is designed to be invoked by a parent MicroBot when the parent determines that a sub-agent is needed to perform a specific task.

    Key characteristics:
    - Runs as a separate process, but shares the environment with the parent MicroBot.
    - Communicates with the parent MicroBot via standard input/output or other IPC mechanisms.
    - Executes commands and performs tasks as directed by the parent MicroBot, but can also have its own internal logic and decision-making capabilities.
    - Can be configured with specific permissions (e.g., read-only or write access) to the repository or environment it operates on.
    """

    name: str = Field(default="MicroBot_sub")
    description: str = Field(default="A sub-agent that can be invoked by the parent MicroBot to perform specific tasks within the same sandbox environment.")
    usage_instructions_to_llm: str = Field(default=INSTRUCTIONS_TO_LLM)

    def is_invoked(self, command: str) -> bool:
        return command.startswith("microbot_sub")

    def invoke(self, command: str, parent_bot: 'MicroBot') -> str:
        # Extract task, iterations, and timeout from the command
        # This is a simplified parsing logic; in a real implementation, you would want to use a more robust method (e.g., argparse)
        parts = command.split("--")
        task = None
        iterations = 25  # default value
        timeout = 300    # default value

        for part in parts:
            if part.startswith("task"):
                task = part.replace("task", "").strip()
            elif part.startswith("iterations"):
                iterations = int(part.replace("iterations", "").strip())
            elif part.startswith("timeout"):
                timeout = int(part.replace("timeout", "").strip())

        if not task:
            logger.error("No task specified for microbot_sub invocation.")
            return "Error: No task specified."

        if iterations <= 0 or timeout <= 0:
            logger.error("Iterations and timeout must be positive integers.")
            return "Error: Iterations and timeout must be positive integers."

        if parent_bot.iteration_count + iterations > parent_bot.max_iterations:
            logger.error("Invoking this sub-agent would exceed the parent bot's maximum iteration count.")
            return "Error: Invoking this sub-agent would exceed the parent bot's maximum iteration count."

        logger.info(f"Invoking MicroBotSubAgent with task: {task}, iterations: {iterations}, timeout: {timeout}")

        sub_bot: MicroBot = MicroBot(
            model=parent_bot.model,
            system_prompt=system_prompt_common,
            environment=parent_bot.environment,
        )

        result: BotRunResult = sub_bot.run(task=task, max_iterations=iterations, timeout_in_seconds=timeout)

        # Not to let LLM bypass the iteration count of the parent bot, we need to subtract the iterations used by the sub-bot from the parent bot's iteration count
        parent_bot.iteration_count -= sub_bot.iteration_count

        if result.status:
            logger.info(f"Sub-agent completed successfully with output: {result.output}")
            return result.output
        else:
            logger.error(f"Sub-agent failed with output: {result.output}\nerror: {result.error}")
            return f"Sub-agent failed with output: {result.output}\nerror: {result.error}"



        # Here you would implement the logic to create and manage the sub-agent process,
        # pass the task to it, and handle its output. For simplicity, we'll just return a placeholder response.

        return f"Sub-agent invoked with task: {task}, iterations: {iterations}, timeout: {timeout}"