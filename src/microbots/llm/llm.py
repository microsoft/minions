from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
from logging import getLogger

logger = getLogger(__name__)


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

    def summarize_context(self, last_n_messages: int = 10, summary: str="") -> dict:
        """
        It is a helper function for the LLM to summarize its own context.
        Leave the last N messages and add the summary between system prompt and the last N messages.

        summary can be empty. If empty, empty summary will be added.
        """
        logger.debug("Messages : %s", self.messages)
        # Keep the system prompt
        msg0 = self.messages[0]["content"]
        # Pop the last message which asked for summarization
        self.messages.pop()
        # Get the last N conversations (user + assistant)
        # If there are not enough messages, take all except system prompt
        if (len(self.messages) > (last_n_messages*2 - 1)):
            logger.debug("Summarizing last %d messages", last_n_messages)
            recent_messages = self.messages[-(last_n_messages*2 - 1):]
        else:
            logger.debug("Not enough messages to summarize, taking all except system prompt")
            recent_messages = self.messages[1:]
        logger.debug("Recent messages that will not be summarized: %s", recent_messages)

        # Update system prompt if it already has a summary
        # summary will be between __summary__ and __end_summary__
        if "__summary__" in msg0:
            system_prompt = msg0.split("__summary__")[0]
            old_summary = msg0.split("__end_summary__")[0].split("__summary__")[1]
            logger.debug("Old summary found: %s", old_summary)
            combined_summary = old_summary + "\n" + summary
        else:
            system_prompt = msg0
            combined_summary = summary

        new_system_prompt = f"{system_prompt}\n__summary__\n{combined_summary}\n__end_summary__"

        # Append without previous user message
        self.messages = [{"role": "system", "content": new_system_prompt}] + recent_messages[:-1]
        logger.debug("Context summarized. New system prompt: %s", new_system_prompt)

        logger.debug("Last message before summarization: %s", recent_messages[-1])
        return recent_messages[-1]  # return the last user message that given before summarization
