from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
from logging import getLogger

logger = getLogger(__name__)

@dataclass
class LLMAskResponse:
    task_done: bool = False
    command: str = ""
    result: str | None = None

llm_output_format_str = """
{
    "task_done": <bool>,  // Indicates if the task is completed
    "command": <str>,     // The command to be executed
    "result": <str|null>  // The result of the command execution, null if not applicable
}
"""

class LLMInterface(ABC):
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
            logger.info("The llm response is %s ", response_dict)

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
                result=response_dict.get("result"),
            )
            return True, llm_response
        else:
            self.retries += 1
            logger.warning("LLM response is missing required fields. Retrying... (%d/%d)", self.retries, self.max_retries)
            self.messages.append({"role": "user", "content": "LLM_RES_ERROR: LLM response is missing required fields. Please respond in the correct JSON format.\n" + llm_output_format_str})
            return False, None
