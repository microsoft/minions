from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
from logging import getLogger

logger = getLogger(__name__)


def _escape_control_chars(s: str) -> str:
    """Escape control characters and fix invalid escape sequences for JSON parsing.
    
    JSON spec issues that LLMs commonly produce:
    1. Unescaped control characters (0x00-0x1F) inside strings - tabs, etc.
    2. Invalid escape sequences like \\& (backslash followed by non-escape char)
    
    We keep newlines as-is since they're part of JSON structure (between fields).
    """
    # Valid JSON escape sequences (after the backslash)
    valid_json_escapes = set('"\\/bfnrtu')
    
    result = []
    i = 0
    while i < len(s):
        char = s[i]
        code = ord(char)
        
        if code < 0x20 and char != '\n':  # Control char but not newline
            if char == '\t':
                result.append('\\t')
            elif char == '\r':
                result.append('\\r')
            else:
                # Escape other control chars as Unicode
                result.append(f'\\u{code:04x}')
        elif char == '\\' and i + 1 < len(s):
            # Check if this backslash is part of a valid JSON escape
            next_char = s[i + 1]
            if next_char in valid_json_escapes:
                # Valid escape sequence, keep both chars and skip ahead
                result.append(char)
                result.append(next_char)
                i += 2
                continue
            else:
                # Invalid escape sequence - double the backslash
                result.append('\\\\')
        elif char == '\\':
            # Trailing backslash with no next char - double it
            result.append('\\\\')
        else:
            result.append(char)
        i += 1
    return ''.join(result)


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
            logger.debug("Attempting to parse JSON response (length: %d)", len(response))
            # Sanitize control characters that may appear unescaped in JSON strings
            # JSON spec forbids unescaped control chars (0x00-0x1F) inside strings
            # LLMs sometimes output literal tabs/etc in command strings
            sanitized_response = _escape_control_chars(response)
            response_dict = json.loads(sanitized_response)
            logger.debug("JSON parsed successfully")
        except json.JSONDecodeError as e:
            self.retries += 1
            logger.warning("LLM response is not valid JSON. Error: %s. Retrying... (%d/%d)", str(e), self.retries, self.max_retries)
            logger.debug("Failed response (repr, first 500 chars): %s", repr(response[:500]) if response else "empty")
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
                thoughts=response_dict.get("thoughts"),
            )
            return True, llm_response
        else:
            self.retries += 1
            logger.warning("LLM response is missing required fields. Retrying... (%d/%d)", self.retries, self.max_retries)
            self.messages.append({"role": "user", "content": "LLM_RES_ERROR: LLM response is missing required fields. Please respond in the correct JSON format.\n" + llm_output_format_str})
            return False, None
