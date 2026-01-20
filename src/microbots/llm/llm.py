from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
from logging import getLogger
from typing import Optional, List

logger = getLogger(__name__)


@dataclass
class DeferredTask:
    """A deferred task with its context summary."""
    task: str
    summary: str = ""  # Summary/context at the point this task was deferred


llm_output_format_str = """
{
    "task_done": <bool>,  // Indicates if the task is completed
    "thoughts": <str>,     // The reasoning behind the decision
    "command": <str>     // The command to be executed
}
"""

@dataclass
class LLMAskResponse:
    task_done: bool = False
    thoughts: str = ""
    command: str = ""

class LLMInterface(ABC):
    def __init__(self, system_prompt: str, max_retries: int = 3):
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.retries = 0
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]
        # Deferred task stack (LIFO) - most recent deferred task is on top
        self.deferred_tasks: List[DeferredTask] = []
        self.llm_reward = 0

    @abstractmethod
    def ask(self, message: str) -> LLMAskResponse:
        pass

    @abstractmethod
    def clear_history(self) -> bool:
        pass

    def _validate_llm_response(self, response: str) -> tuple[bool, LLMAskResponse]:

        if self.retries >= self.max_retries:
            logger.error("Maximum retries reached for LLM response validation.")
            raise Exception("LLM is not responding in expected format. Maximum retries reached.")

        try:
            response_dict = json.loads(response)
        except json.JSONDecodeError:
            self.retries += 1
            logger.warning("LLM response is not valid JSON. Retrying... (%d/%d)", self.retries, self.max_retries)
            self.messages.append({"role": "user", "content": "LLM_RES_ERROR: Please respond in the correct JSON format.\n" + llm_output_format_str})
            return False, None

        if all(key in response_dict for key in LLMAskResponse.__annotations__.keys()):
            logger.debug("The llm response is %s ", response_dict)

            if response_dict.get("task_done") not in [True, False]:
                self.retries += 1
                logger.warning("LLM response 'task_done' field is not a boolean. Retrying... (%d/%d)", self.retries, self.max_retries)
                self.messages.append({"role": "user", "content": "LLM_RES_ERROR: Please ensure 'task_done' is a boolean (true/false).\n" + llm_output_format_str})
                return False, None

            if (
                response_dict.get("task_done") is False
                and (
                    response_dict.get("command") is None
                    or not isinstance(response_dict.get("command"), str)
                    or response_dict.get("command").strip() == ""
                    )
            ):
                self.retries += 1
                logger.warning("LLM response 'command' field is invalid. Retrying... (%d/%d)", self.retries, self.max_retries)
                self.messages.append({"role": "user", "content": "LLM_RES_ERROR: Please ensure 'command' is a non-empty string.\n" + llm_output_format_str})
                return False, None

            if (response_dict.get("task_done") is True):
                command = response_dict.get("command", None)
                if command is not None and command.strip() != "":
                    self.retries += 1
                    logger.warning("LLM response 'command' should be empty when 'task_done' is true. Retrying... (%d/%d)", self.retries, self.max_retries)
                    self.messages.append({"role": "user", "content": "LLM_RES_ERROR: When 'task_done' is true, 'command' should be an empty string.\nYou should set 'task_done' to true only when even the last command got executed successfully.\nExpected output format:\n" + llm_output_format_str})
                    return False, None

            llm_response = LLMAskResponse(
                task_done=response_dict["task_done"],
                command=response_dict["command"],
                thoughts=response_dict.get("thoughts"),
            )
            return True, llm_response
        else:
            self.retries += 1
            logger.warning("LLM response is missing required fields. Retrying... (%d/%d)", self.retries, self.max_retries)
            self.messages.append({"role": "user", "content": "LLM_RES_ERROR: LLM response is missing required fields. Please respond in the correct JSON format.\n" + llm_output_format_str})
            return False, None

    def update_context(self, last_n_messages: int = 10, summary: str="") -> dict:
        """
        It is a helper function for the LLM to summarize its own context.
        Leave the last N messages and add the summary between system prompt and the last N messages.

        summary can be empty. If empty, empty summary will be added.
        """
        logger.debug("Messages : %s", self.messages)

        # Keep the system prompt
        if self.messages[0]["role"] == "system":
            msg0 = self.messages[0]["content"]
        else:
            msg0 = self.system_prompt

        # Pop the last message which asked for summarization
        self.messages.pop()
        # Get the last N conversations (user + assistant)
        # If there are not enough messages, take all except system prompt
        if last_n_messages == 0:
            logger.debug("last_n_messages is 0. Summarizing all conversations so far.")
            recent_messages = []
        elif (len(self.messages) > (last_n_messages*2)):
            # [s, u1, a1, u2, a2, u3, a3]
            #
            # last_n_messages = 1
            # len(messages) = 7 > 2
            # recent_messages = messages[-2:] => [u3, a3]
            #
            # last_n_messages = 2
            # len(messages) = 7 > 4
            # recent_messages = messages[-4:] => [u2, a2, u3, a3]
            #
            # last_n_messages = 3
            # len(messages) = 7 > 6
            # recent_messages = messages[-6:] => [u1, a1, u2, a2, u3, a3]
            logger.debug("Summarizing first %d conversations", (len(self.messages) - last_n_messages))
            recent_messages = self.messages[-(last_n_messages*2):]
        else:
            logger.debug("Not enough messages to summarize, taking all except system prompt")
            recent_messages = self.messages[1:]
        logger.debug("Recent messages that will not be summarized: %s", recent_messages)

        # Update system prompt if it already has a summary
        # summary will be between __context__ and __end_context__
        if "__context__" in msg0:
            system_prompt = msg0.split("__context__")[0]
            # old_summary = msg0.split("__end_context__")[0].split("__context__")[1]
            # logger.debug("Old summary found: %s", old_summary)
            # combined_summary = old_summary + "\n" + summary
        else:
            system_prompt = msg0
            # combined_summary = summary

        combined_summary = summary

        new_system_prompt = f"{system_prompt}\n\n__context__\n{combined_summary}\n__end_context__"

        if recent_messages:
            # Append without previous user message
            self.messages = [{"role": "system", "content": new_system_prompt}] + recent_messages[:-1]
            logger.debug("Context summarized. New system prompt: %s", new_system_prompt)

            logger.debug("Last message before summarization: %s", recent_messages[-1])
            return recent_messages[-1]["content"]  # return the last user message that given before summarization

        else:
            # No recent messages, just keep system prompt
            self.messages = [{"role": "system", "content": new_system_prompt}]
            logger.debug("No recent messages to keep after summarization. New system prompt: %s", new_system_prompt)
            return "Context is updated as per your request.\nIf you think the current task is complete, set task_done to true. Deferred tasks (if any) will be given to you next."

    def do_later(self, current_task: str, deferred_task: str) -> str:
        """
        Defer a sub-task to later and focus on the current task.

        This method:
        1. Pushes the deferred_task onto the stack
        2. Resets the conversation history
        3. Updates the system prompt with current_task in place of summary
        4. Returns a message for the LLM to start the current task

        Args:
            current_task: The immediate sub-task to work on now
            deferred_task: The remaining task(s) to defer for later

        Returns:
            str: A user message to present to the LLM for starting current_task
        """
        self.llm_reward += 1
        logger.debug("do_later called with current_task: %s, deferred_task: %s", current_task, deferred_task)

        # Get current context summary before reset
        current_summary = ""
        if "__context__" in self.system_prompt:
            current_summary = self.system_prompt.split("__end_context__")[0].split("__context__")[1].strip()

        # Push deferred task onto stack
        self.deferred_tasks.append(DeferredTask(
            task=deferred_task,
            summary=current_summary
        ))
        logger.debug("Deferred task pushed onto stack. Stack size: %d", len(self.deferred_tasks))

        # Get base system prompt (strip any existing summary)
        base_system_prompt = self.system_prompt.split("__context__")[0].rstrip()

        # Update system prompt with current_task as the new context
        new_system_prompt = f"{base_system_prompt}\n__context__\n<current_task>\n{current_task}\n</current_task>\n__end_context__"

        # Replace line with # $$REWARD$$: <reward>
        if "# $$REWARD$$:" in new_system_prompt:
            new_system_prompt = "\n".join([
                line if not line.startswith("# $$REWARD$$:") else f"# $$REWARD$$: {self.llm_reward}"
                for line in new_system_prompt.splitlines()
            ])

        self.system_prompt = new_system_prompt
        self.messages = [
            {
                "role": "system",
                "content": new_system_prompt,
            }
        ]

        logger.debug("Context reset with current_task. New system prompt: %s", new_system_prompt[:500])

        # Return message to start the current task
        # return f"START TASK: {current_task}\n\nNote: You have deferred other task(s) for later. Focus only on this task now. When done, set task_done to true."
        return current_task

    def complete_current_and_get_next(self) -> Optional[str]:
        """
        Pop the next deferred task from the stack.

        This method is called when task_done is True.
        It:
        1. Pops the most recent deferred task from the stack
        2. Resets llm.messages
        3. Returns the deferred task as a new user message with its context

        Returns:
            Optional[str]: The deferred task message, or None if stack is empty
        """
        logger.debug("complete_current_and_get_next called. Stack size: %d", len(self.deferred_tasks))

        if not self.deferred_tasks:
            logger.debug("No deferred tasks in stack")
            return None

        # Pop the most recent deferred task
        next_task = self.deferred_tasks.pop()
        logger.debug("Popped deferred task: %s. Remaining stack size: %d", next_task.task, len(self.deferred_tasks))

        # Get base system prompt (strip any existing summary)
        if "__context__" in self.system_prompt:
            base_system_prompt = self.system_prompt.split("__context__")[0].rstrip()
        else:
            base_system_prompt = self.system_prompt

        current_summary = ""
        if "__context__" in self.system_prompt:
            current_summary = self.system_prompt.split("__end_context__")[0].split("__context__")[1].strip()

        # Add context about completed work and next task
        completed_summary = ""
        if current_summary:
            completed_summary = f"{current_summary}\nStatus: COMPLETE"
        else:
            completed_summary = "Previous sub-task: COMPLETE" # Impossible case

        if len(self.messages) > 1:
            completed_summary += "\n\n<conversation>\n" + "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.messages[1:]]) + "\n</conversation>"

        if next_task.summary:
            completed_summary += f"\n\n---\nContext from when this task was deferred:\n{next_task.summary}"

        new_system_prompt = f"{base_system_prompt}\n__context__\n{completed_summary}\n__end_context__" + "\n\nBased on current state of the project, first update your context with all necessary information. Then either use `do_later` to take a sub-task or set task_done to true if all tasks are complete.\nYOU SHOULD BE VERY CAREFUL WHILE MANAGING CONTEXT FOR CURRENT TASK AND DEFERRED TASKS."

        # Replace line with # $$REWARD$$: <reward>
        if "$$REWARD$$:" in new_system_prompt:
            new_system_prompt = "\n".join([
                line if not line.startswith("$$REWARD$$:") else f"$$REWARD$$: {self.llm_reward}"
                for line in new_system_prompt.splitlines()
            ])

        self.system_prompt = new_system_prompt

        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

        logger.debug("Messages reset for deferred task: %s", next_task.task)

        # Return the deferred task as a new user message
        # context_info = next_task.summary if next_task.summary else "Starting fresh on this deferred task."
        return f"DEFERRED TASK (continue from where you left off):\n\n{next_task.task}"
