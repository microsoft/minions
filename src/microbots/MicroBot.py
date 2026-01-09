import json
from pprint import pformat
import re
import time
from dataclasses import dataclass
from enum import StrEnum
from logging import getLogger
from typing import Optional

from microbots.constants import ModelProvider
from microbots.environment.local_docker.LocalDockerEnvironment import (
    LocalDockerEnvironment,
)
from microbots.llm.anthropic_api import AnthropicApi
from microbots.llm.openai_api import OpenAIApi
from microbots.llm.ollama_local import OllamaLocal
from microbots.llm.llm import llm_output_format_str
from microbots.tools.tool import Tool, install_tools, setup_tools
from microbots.extras.mount import Mount, MountType
from microbots.utils.logger import LogLevelEmoji, LogTextColor
from microbots.utils.network import get_free_port

logger = getLogger(" MicroBot ")

system_prompt_common_backup = f"""
You are an intelligent agent well versed in software development and debugging.

You will be provided with a coding or debugging task to complete inside a sandboxed shell environment.
You will break down the task into smaller steps and use your reasoning skills to complete the task using shell commands.
There is a shell session open for you. You will be provided with a task and you should achieve it using the shell commands.
All your response must be in the following json format:
{llm_output_format_str}
The properties ( task_done, thoughts, command ) are mandatory on each response.
Give the command one at a time to solve the given task. As long as you're not done with the task, set task_done to false.
When you are sure that the task is completed, summarize your steps along with your final thoughts. Then set task_done to true, set command to empty string.
Don't add any chat or extra messages outside the json format. Because the system will parse only the json response.
Any of your thoughts must be in the 'thoughts' field.

after each command, the system will execute the command and respond to you with the output.
Ensure to run only one command at a time.
NEVER use commands that produce large amounts of output or take a long time to run to avoid exceeding context limits.
Use specific patterns: 'find <path> -name "*.c" -maxdepth 2' instead of recursive exploration.
No human is involved in the task. So, don't seek human intervention.

Use Chain of Thought to break down the steps into smaller sub-tasks. And use `do_later` tool to defer sub-tasks other than most immediate one.
Reading and understanding the code should be considered as a sub-task. It should not be mixed with code modification or writing tasks.
When you have to update multiple locations of a file, you can even use `do_later` to set the current task as editing one location and defer the rest of the locations to later.
Every time you break down a sub-task using `do_later`, you'll be given a reward. You can see your reward below at "$$REWARD$$". You should try to maximize your reward.

Remember following important points
1. If a command fails, analyze the error message and provide an alternative command in your next response. Same command will not pass again.
2. Avoid using recursive commands like 'ls -R', 'rm -rf', 'tree', or 'find' without depth limits as they can produce excessive output or be destructive.
3. You cannot run any interactive commands like vim, nano, etc.

# TOOLS
You have following special tools.

    1. summarize_context: Use this tool if your input has too many irrelevant conversation turns.
         You can use this tool to rewrite your own context in a concise manner focusing on important points only. For example, if you have a failed command output which you've solved in later steps deep down in the conversation, that is not required to be in the context. You can summarize the context to remove such irrelevant information. This tool will not update the system prompt. So, you can ignore details in the system prompt while summarizing.

         Usage:
            summarize_context <no_of_recent_turns_to_keep> "<your summary of the context>"

            <no_of_recent_turns_to_keep> : Number of recent conversation turns to keep as is without summarizing. 1 means last user-assistant pair will be kept as is.
            "<your summary of the context>" : Your summarized context in double quotes. The summary can be empty if you finished a sub-task and want to remove previous context.

        Important Notes:
            - The summarize tool call step will not be added to your history.
            - Try to be very precise and concise in your summary.
            - Always use this tool before marking a task as done. Because, the next task may depend on the current task's context.

    2. do_later: Use this tool to defer a sub-task to later.
         This tool gives you an indirect way to plan and break down your tasks into smaller sub-tasks.
         You should always plan a task into multiple steps and use this tool to set your immediate sub-task as current task and defer the rest to later.
         When you complete your current sub-task, the system will automatically give you the deferred task as your next task along with the context of already completed work.
         A wise use of this tool will help you to keep your context light and focused on the immediate task only.

         Usage:
            do_later "<current_task>" "<deferred_task>"

            "<current_task>" : The immediate sub-task you are going to start working on now. Along, with context of already completed work.
            "<deferred_task>" : Broken down down or cumulative version of remaining task you want to defer to later but you don't want clutter your current context with it.

        Important Notes:
            - Your summary will be replaced with <current_task> and all your conversation history will be cleared except the system prompt.
            - If you want to maintain some context from previous summary or conversation, include that in the <current_task>.
            - The <deferred_task> will be given to you when you complete the current task. You don't need to remember it. The system will take care of it.
            - It is prudent to take very minimal sub-task as <current_task> to keep your context light.
            - You don't need to break down the entire remaining task in <deferred_task>. A cumulative version is sufficient.

# $$REWARD$$: 0
"""

system_prompt_common = f"""
# ROLE
You are an expert software engineer and debugging specialist.
You plan, split tasks and complete them in a systematic manner.

# OBJECTIVE
Complete the assigned coding or debugging task by executing shell commands step-by-step.
Think methodically, break complex tasks into manageable sub-tasks, and execute one command at a time.

# RESPONSE FORMAT
All responses MUST be valid JSON in this exact format:
{llm_output_format_str}

**Required Fields:**
- `task_done`: boolean - Set to `false` while working, `true` only when task is fully complete
- `thoughts`: string - Your reasoning, observations, and next steps
- `command`: string - Single shell command to execute (empty string when task_done is true)

⚠️ Output ONLY the JSON object. No additional text, explanations, or markdown outside the JSON structure.

# EXECUTION WORKFLOW
1. Receive task → Analyze requirements
2. Break down into first vs rest using `do_later` tool
3. Execute one command → Observe output → Reason about next step
4. Repeat until task complete
5. Summarize work using `summarize_context` and set `task_done: true`

# CRITICAL RULES
1. Always start by breaking down the task using `do_later` to focus on immediate sub-task.
2. Break reading/understanding code into separate sub-task from code modification/writing. The first task should be exploring/reading only and update its context. The second task will immediately start updating code based on the context from first task.
3. At the end of each sub-task, use `summarize_context` to condense your context. There may be a next task that will continue from where you left off. So, include all necessary details like file names, function names, line numbers, error messages, even code-snippets, etc. in your summary.

## Command Execution
- Execute ONE command per response
- NEVER use interactive commands (vim, nano, less, etc.)
- AVOID commands with large output - use specific patterns:
  ✓ `find <path> -name "*.c" -maxdepth 2`
  ✗ `find / -name "*.c"` (unbounded)
  ✗ `ls -R`, `tree` without limits
- If a command fails, analyze the error and try an alternative approach. Repeating the same command will fail again.

## Task Management
- Use Chain of Thought reasoning to decompose complex tasks
- Keep reading/understanding code SEPARATE from modifying code. For example, keep the actual task in deferred context and set current task to exploring/reading code.
- When modifying multiple locations in a file, use `do_later` to handle one location at a time
- No human intervention available - resolve all issues autonomously

# TOOLS

## 1. summarize_context
Condense your conversation history to stay within context limits and maintain focus.

Provide necessary details in the summary to ensure continuity of the task.

If you're completing a task, summarize all the conversations by setting turns_to_keep to 0 and mention in your summary that the sub-task is completed and the LLM should simply set task_done to true in the next step.

**Syntax:**
```
summarize_context <turns_to_keep> "<summary>"
```

**Parameters:**
- `<turns_to_keep>`: Number of recent user-assistant exchanges to preserve (1 = last exchange)
- `<summary>`: Concise summary of important context (can be just the task name and it's status as complete). If it is a context gathering job, include all necessary details like file names, function names, line numbers, error messages, even code-snippets, etc. in your summary.

**Best Practices:**
- Use BEFORE marking task_done to clean up context for next task
- Use it in-between tasks when you have multiple failed attempts. Copy all the previous conversations that are necessary for context into the context summary.
- Use it to validate your changes. Once you've made all the changes, clear-up the conversation and set the context as task is complete it time to read and validate.
- Once you validate successfully, use it to summarize the entire work done so far including validation details and mention only task_done is pending to be set to true in next step.

## 2. do_later
It is a novel way of planning. Instead of breaking a task into multiple sub-tasks, you can use this tool to set your immediate sub-task as current task and defer the rest to later.
The deferred tasks will be automatically provided to you when you complete the current task. You can again use do_later to take the next immediate sub-task and defer the rest. It will create a tree of tasks and sub-tasks which will be handled recursively.

**Syntax:**
```
do_later "<current_task>" "<deferred_task>"
```

**Parameters:**
- `<current_task>`: Immediate sub-task to work on now (include any essential context from prior work)
- `<deferred_task>`: Remaining work to handle after current task completes. Include all necessary context as part of this task. Be as elaborate as possible. It is important to keep the known context with the deferred task because the subtasks in current chain may override the context summary anytime. Context summary is volatile as you can always change it using `summarize_context`. So, it is prudent to keep the necessary context with the deferred task.

**Behavior:**
- Your context resets to system prompt + `<current_task>` only
- `<deferred_task>` is queued and automatically provided upon completion
- No need to remember deferred tasks - the system tracks them

**Strategy:**
- Take the SMALLEST viable sub-task as current
- Deferred task can be a cumulative description (no need to fully decompose)

# REWARD SYSTEM
$$REWARD$$: 0
Maximize your reward by effectively using `do_later` to break down tasks!

# Example flow:
If you are tasked to backport a patch, you should do following tasks in order
 - Use `do_later` to set current task as exploring the codebase to understand the patch and its dependencies. Defer the actual backporting as deferred task. Mention in the current task to update the context with findings in detail with file names, functions and even code snippet. When you've explored the upstream patch, copy it to the context summary. Other tasks will require it.
 - If your task is to understand a patch for backporting purposes, read relevant code and update your context summary with detailed findings using `summarize_context`. Mention in that summary all the necessary details required for backporting like the upstream patch content, files locations, function names, etc., in the target branch. You are welcome to include code snippets in the summary. You should use a structured format for easy understanding.
 - If you got the summary from the first task and now you are actually backporting the patch, set the current task to backport the first hunk of the patch. Defer the rest of the patch backporting as deferred task. Reset the context with summary from previous task and snippet of the hunk to be backported. Keep the full context with the deferred task.
 - Keep doing until the entire patch is backported.
 - At the end, clear all the conversations and just set the context to verify the backporting is successful by reading the code.
"""

class BotType(StrEnum):
    READING_BOT = "READING_BOT"
    WRITING_BOT = "WRITING_BOT"
    BROWSING_BOT = "BROWSING_BOT"
    CUSTOM_BOT = "CUSTOM_BOT"
    LOG_ANALYSIS_BOT = "LOG_ANALYSIS_BOT"


@dataclass
class BotRunResult:
    status: bool
    result: str | None
    error: Optional[str]


class MicroBot:
    """
    The core Microbot class.

    MicroBot class is the core class representing the autonomous agent. Other bots are extensions of this class.
    If you want to create a custom bot, you can directly use this class or extend it into your own bot class.

    Attributes
    ----------
        model : str
            The model to use for the bot, in the format <provider>/<model_name>.
        bot_type : BotType
            The type of bot being created. It's unused. Will be removed soon.
        system_prompt : Optional[str]
            The system prompt to guide the bot's behavior.
        environment : Optional[any]
            The execution environment for the bot. If not provided, a default
            LocalDockerEnvironment will be created.
        additional_tools : Optional[list[Tool]]
            A list of additional tools to install in the bot's environment.
        folder_to_mount : Optional[Mount]
            A folder to mount into the bot's environment. The bot will be given
            access to this folder based on the specified permissions. This will
            be the main code folder where the bot will work. Additional folders
            can be mounted during the run() method. Refer to `Mount` class
            regarding the directory structure and permission details. Defaults
            to None.
    """

    def __init__(
        self,
        model: str,
        bot_type: BotType = BotType.CUSTOM_BOT,
        system_prompt: Optional[str] = None,
        environment: Optional[any] = None,
        additional_tools: Optional[list[Tool]] = [],
        folder_to_mount: Optional[Mount] = None,
    ):
        """
        Init function for MicroBot class.

        Parameters
        ----------
            model :str
                The model to use for the bot, in the format <provider>/<model_name>.
            bot_type :BotType
                The type of bot being created. It's unused. Will be removed soon.
            system_prompt :Optional[str]
                The system prompt to guide the bot's behavior. Defaults to None.
            environment :Optional[any]
                The execution environment for the bot. If not provided, a default
                LocalDockerEnvironment will be created.
            additional_tools :Optional[list[Tool]]
                A list of additional tools to install in the bot's environment.
                Defaults to [].
            folder_to_mount :Optional[Mount]
                A folder to mount into the bot's environment. The bot will be given
                access to this folder based on the specified permissions. This will
                be the main code folder where the bot will work. Additional folders
                can be mounted using the run() method. Refer to `Mount` class
                regarding the directory structure and permission details. Defaults
                to None.

                Note: Supports only mount type MountType.MOUNT for now.
        """

        self.folder_to_mount = folder_to_mount

        # TODO : Need to check on the purpose of variable `mounted`
        # 1. If we allow user to mount multiple directories,
        # we should able to get it as an argument and store them in self.mounted.
        # This require changes in _create_environment to handle multiple mount directories or files.
        #
        # 2. We should let user to mount only one directory. In that case self.mounted may not be required.
        # Just one self.folder_to_mount and necessary extra mounts at the derived class similar to LogAnalyticsBot.

        self.mounted = []
        if folder_to_mount is not None:
            self._validate_folder_to_mount(folder_to_mount)
            self.mounted.append(folder_to_mount)

        self.system_prompt = system_prompt
        self.model = model
        self.bot_type = bot_type
        self.environment = environment
        self.additional_tools = additional_tools

        self._validate_model_and_provider(model)
        self.model_provider = model.split("/")[0]
        self.deployment_name = model.split("/")[1]

        if not self.environment:
            self._create_environment(self.folder_to_mount)

        self._create_llm()

        install_tools(self.environment, self.additional_tools)

    def run(
        self,
        task: str,
        additional_mounts: Optional[list[Mount]] = None,
        max_iterations: int = 20,
        timeout_in_seconds: int = 200
    ) -> BotRunResult:

        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than 0")

        setup_tools(self.environment, self.additional_tools)

        for mount in additional_mounts or []:
            self._mount_additional(mount)

        iteration_count = 1
        # start timer
        start_time = time.time()
        timeout = timeout_in_seconds
        llm_response = self.llm.ask(task)
        return_value = BotRunResult(
            status=False,
            result=None,
            error="Did not complete",
        )
        logger.info("%s TASK STARTED : %s...", LogLevelEmoji.INFO, task[0:60])

        while llm_response.task_done is False:
            logger.info("%s Step-%d %s", "-" * 20, iteration_count, "-" * 20)
            if llm_response.thoughts:
                logger.info(
                    f" 💭  LLM thoughts: {LogTextColor.OKCYAN}{llm_response.thoughts}{LogTextColor.ENDC}",
                )
            logger.info(
                f" ➡️  LLM tool call : {LogTextColor.OKBLUE}{llm_response.command}{LogTextColor.ENDC}",
            )
            # increment iteration count
            iteration_count += 1
            if iteration_count >= max_iterations:
                return_value.error = f"Max iterations {max_iterations} reached"
                return return_value

            # check if timeout has reached
            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time > timeout:
                logger.error(
                    "Iteration %d with response %s - Exiting without running command as timeout reached",
                    iteration_count,
                    llm_response,
                )
                return_value.error = f"Timeout of {timeout} seconds reached"
                return return_value

            # Handle context summarization
            if llm_response.command.startswith("summarize_context"):
                parsed_args = self._parse_summarize_context_command(llm_response.command)
                if parsed_args is None:
                    # Invalid syntax - ask LLM to correct it
                    error_msg = self._get_summarize_context_syntax_error(llm_response.command)
                    llm_response = self.llm.ask(error_msg)
                    continue
                last_n_messages, summary = parsed_args
                last_msg = self.llm.summarize_context(last_n_messages=last_n_messages, summary=summary)
                llm_response = self.llm.ask(last_msg["content"])
                continue

            # Handle do_later command
            if llm_response.command.startswith("do_later"):
                parsed_args = self._parse_do_later_command(llm_response.command)
                if parsed_args is None:
                    # Invalid syntax - ask LLM to correct it
                    error_msg = self._get_do_later_syntax_error(llm_response.command)
                    llm_response = self.llm.ask(error_msg)
                    continue
                current_task, deferred_task = parsed_args
                new_user_msg = self.llm.do_later(current_task=current_task, deferred_task=deferred_task)
                llm_response = self.llm.ask(new_user_msg)
                continue

            # Validate command for dangerous operations
            is_safe, explanation = self._is_safe_command(llm_response.command)
            if not is_safe:
                error_msg = f"Dangerous command detected and blocked: {llm_response.command}\n{explanation}"
                logger.info("%s %s", LogLevelEmoji.WARNING, error_msg)
                llm_response = self.llm.ask(f"COMMAND_ERROR: {error_msg}\nPlease provide a safer alternative command.")
                continue

            llm_command_output = self.environment.execute(llm_response.command)

            logger.debug(
                    " 🔧  Command executed.\nReturn Code: %d\nStdout:\n%s\nStderr:\n%s",
                    llm_command_output.return_code,
                    llm_command_output.stdout,
                    llm_command_output.stderr,
                )

            if llm_command_output.return_code == 0:
                if llm_command_output.stdout:
                    output_text = llm_command_output.stdout
                    # HACK: anthropic-text-editor tool extra formats the output
                    try:
                        output_json = json.loads(llm_command_output.stdout)
                        if isinstance(output_json, dict) and "content" in output_json:
                            output_text = pformat(output_json["content"])
                    except json.JSONDecodeError:
                        pass
                else:
                    output_text = f"Command executed successfully with no output\nreturn code: {llm_command_output.return_code}\nstdout: {llm_command_output.stdout}\nstderr: {llm_command_output.stderr}"
            else:
                output_text = f"COMMAND EXECUTION FAILED\nreturn code: {llm_command_output.return_code}\nstdout: {llm_command_output.stdout}\nstderr: {llm_command_output.stderr}"

            logger.info(" ⬅️  Command output:\n%s", output_text)
            llm_response = self.llm.ask(output_text)

        if llm_response.thoughts:
            logger.info(
                f" 💭  LLM final thoughts: {LogTextColor.OKCYAN}{llm_response.thoughts}{LogTextColor.ENDC}",
            )
        logger.info("🔚 TASK COMPLETED : %s...", task[0:15])

        # Check if there are deferred tasks to process
        next_deferred_task_msg = self.llm.complete_current_and_get_next()
        if next_deferred_task_msg:
            logger.info("%s Processing deferred task recursively...", LogLevelEmoji.INFO)
            # Recursively call run() with the deferred task
            # This resets context for each branch and handles the tree naturally
            return self.run(
                task=next_deferred_task_msg,
                additional_mounts=None,  # Already mounted
                max_iterations=max_iterations,
                timeout_in_seconds=timeout_in_seconds
            )

        return BotRunResult(status=True, result=llm_response.thoughts, error=None)

    def _mount_additional(self, mount: Mount):
        if mount.mount_type != MountType.COPY:
            logger.error(
                "%s Only COPY mount type is supported for additional mounts for now",
                LogLevelEmoji.ERROR,
            )
            raise ValueError(
                "Only COPY mount type is supported for additional mounts for now"
            )

        self.mounted.append(mount)
        copy_to_container_result = self.environment.copy_to_container(
            mount.host_path_info.abs_path, mount.sandbox_path
        )
        if copy_to_container_result is False:
            raise ValueError(
                f"Failed to copy additional mount to container: {mount.host_path_info.abs_path} -> {mount.sandbox_path}"
            )

    # TODO : pass the sandbox path
    def _create_environment(self, folder_to_mount: Optional[Mount]):
        free_port = get_free_port()

        self.environment = LocalDockerEnvironment(
            port=free_port,
            folder_to_mount=folder_to_mount,
        )

    def _create_llm(self):
        # Append tool usage instructions to system prompt
        system_prompt_with_tools = self.system_prompt if self.system_prompt else ""
        if self.additional_tools:
            for tool in self.additional_tools:
                if tool.usage_instructions_to_llm:
                    system_prompt_with_tools += f"\n\n{tool.usage_instructions_to_llm}"

        if self.model_provider == ModelProvider.OPENAI:
            self.llm = OpenAIApi(
                system_prompt=system_prompt_with_tools, deployment_name=self.deployment_name
            )
        elif self.model_provider == ModelProvider.OLLAMA_LOCAL:
            self.llm = OllamaLocal(
                system_prompt=system_prompt_with_tools, model_name=self.deployment_name
            )
        elif self.model_provider == ModelProvider.ANTHROPIC:
            self.llm = AnthropicApi(
                system_prompt=system_prompt_with_tools, deployment_name=self.deployment_name
            )
        # No Else case required as model provider is already validated using _validate_model_and_provider

    def _validate_model_and_provider(self, model):
        # Ensure it has only only slash
        if model.count("/") != 1:
            raise ValueError("Model should be in the format <provider>/<model_name>")
        provider = model.split("/")[0]
        if provider not in [e.value for e in ModelProvider]:
            raise ValueError(f"Unsupported model provider: {provider}")

    def _validate_folder_to_mount(self, folder_to_mount: Mount):
        if folder_to_mount.mount_type != MountType.MOUNT:
            logger.error(
                "%s Only MOUNT mount type is supported for folder_to_mount",
                LogLevelEmoji.ERROR,
            )
            raise ValueError(
                "Only MOUNT mount type is supported for folder_to_mount"
            )

    def _parse_summarize_context_command(self, command: str) -> tuple[int, str] | None:
        """
        Parse the summarize_context command and extract arguments.

        Expected format: summarize_context <n> "<summary>"
        Where <n> is an integer and <summary> is a quoted string.

        Returns:
            tuple[int, str]: (last_n_messages, summary) if valid
            None: if invalid syntax
        """
        import shlex
        try:
            # Use shlex to properly handle quoted strings
            parts = shlex.split(command)
            if len(parts) < 2 or len(parts) > 3:
                return None

            # First part should be 'summarize_context'
            if parts[0] != "summarize_context":
                return None

            # Second part should be an integer
            try:
                last_n_messages = int(parts[1])
            except ValueError:
                return None

            # Third part is the summary (optional, defaults to empty string)
            summary = parts[2] if len(parts) > 2 else ""

            return (last_n_messages, summary)

        except ValueError:
            # shlex.split can raise ValueError for malformed strings
            return None

    def _get_summarize_context_syntax_error(self, command: str) -> str:
        """
        Generate an error message for invalid summarize_context syntax.

        Returns a detailed error message guiding the LLM to use correct syntax.
        """
        return f"""COMMAND_ERROR: Invalid summarize_context syntax.
Your command: {command}

Correct usage:
    summarize_context <no_of_recent_turns_to_keep> "<your summary of the context>"

Examples:
    summarize_context 2 "Summary of previous work: explored files and found the bug"
    summarize_context 0 ""
    summarize_context 5 "Completed file exploration, ready to make changes"

Please send the command again with correct syntax."""

    def _parse_do_later_command(self, command: str) -> tuple[str, str] | None:
        """
        Parse the do_later command and extract arguments.

        Expected format: do_later "<current_task>" "<deferred_task>"
        Both arguments are quoted strings.

        Returns:
            tuple[str, str]: (current_task, deferred_task) if valid
            None: if invalid syntax
        """
        import shlex
        try:
            # Use shlex to properly handle quoted strings
            parts = shlex.split(command)
            if len(parts) != 3:
                return None

            # First part should be 'do_later'
            if parts[0] != "do_later":
                return None

            # Second part is current_task
            current_task = parts[1]
            if not current_task or current_task.strip() == "":
                return None

            # Third part is deferred_task
            deferred_task = parts[2]
            if not deferred_task or deferred_task.strip() == "":
                return None

            return (current_task, deferred_task)

        except ValueError:
            # shlex.split can raise ValueError for malformed strings
            return None

    def _get_do_later_syntax_error(self, command: str) -> str:
        """
        Generate an error message for invalid do_later syntax.

        Returns a detailed error message guiding the LLM to use correct syntax.
        """
        return f"""COMMAND_ERROR: Invalid do_later syntax.
Your command: {command}

Correct usage:
    do_later "<current_task>" "<deferred_task>"

Examples:
    do_later "Fix the null pointer exception in auth.py" "After fixing auth.py, update the unit tests and documentation"
    do_later "Explore the codebase structure" "Implement the feature after understanding the architecture"
    do_later "Install required dependencies" "Configure the database connection and run migrations"

Important:
    - Both arguments must be non-empty quoted strings
    - <current_task> is what you'll work on immediately
    - <deferred_task> is what will be given to you after completing current_task

Please send the command again with correct syntax."""

    def _get_dangerous_command_explanation(self, command: str) -> Optional[str]:
        """Provides detailed explanation for why a command is dangerous and suggests alternatives.

        Args:
            command: The shell command to analyze

        Returns:
            str: Explanation with reason and alternative, or None if command is safe
        """
        # Handle invalid commands (empty, None, or non-string)
        if not command or not isinstance(command, str):
            return "REASON: Empty or invalid command provided\nALTERNATIVE: Provide a valid shell command"

        stripped_command = command.strip()
        if not stripped_command:
            return "REASON: Empty or whitespace-only command provided\nALTERNATIVE: Provide a valid shell command"

        # Define dangerous command patterns with detailed explanations
        # Note: Don't convert to lowercase before checking, as we need case-sensitive pattern matching
        dangerous_checks = [
            {
                'pattern': r'\bls\s+(?:[^-]*\s+)?-[a-z]*[rR](?:[a-z]*\b|\s|$)',
                'reason': 'Recursive ls commands (ls -R) can generate massive output in large repositories, exceeding context limits',
                'alternative': 'Use targeted paths like "ls drivers/block/" or "ls -la <specific-directory>" instead'
            },
            {
                'pattern': r'\btree\b',
                'reason': 'Tree command recursively lists entire directory structures, which can exceed context limits',
                'alternative': 'Use "ls -la <specific-directory>" or "find <path> -maxdepth 2 -type d" for controlled exploration'
            },
            {
                'pattern': r'\brm\s+(?:[^-]*\s+)?-[a-z]*[rR](?:[a-z]*\b|\s|$)',
                'reason': 'Recursive rm commands (rm -r/-rf) can delete entire directory trees, which is destructive',
                'alternative': 'Delete specific files individually or use "rm <specific-file>" to avoid accidental data loss'
            },
            {
                'pattern': r'\brm\s+--recursive\b',
                'reason': 'Recursive rm commands can delete entire directory trees, which is destructive',
                'alternative': 'Delete specific files individually or use "rm <specific-file>" to avoid accidental data loss'
            },
            {
                'pattern': r'\bfind\b(?!.*-maxdepth)',
                'reason': 'Find command without -maxdepth can recursively search entire filesystems, causing excessive output',
                'alternative': 'Use "find <path> -name "*.ext" -maxdepth 2" to limit search depth and control output size'
            },
        ]

        for check in dangerous_checks:
            if re.search(check['pattern'], stripped_command, re.IGNORECASE):
                return f"REASON: {check['reason']}\nALTERNATIVE: {check['alternative']}"

        return None

    def _is_safe_command(self, command: str) -> tuple[bool, Optional[str]]:
        """Validates if a command is safe to execute.

        A command is considered safe if it:
        - Is not a recursive command (ls -R, rm -rf, tree, find without -maxdepth)
        - Does not risk generating excessive output or destructive actions

        Args:
            command: The shell command to validate

        Returns:
            tuple[bool, Optional[str]]: A tuple of (is_safe, explanation) where:
                - is_safe: True if command is safe to execute, False otherwise
                - explanation: Detailed explanation if dangerous, None if safe
        """
        explanation = self._get_dangerous_command_explanation(command)
        is_safe = explanation is None
        return is_safe, explanation
