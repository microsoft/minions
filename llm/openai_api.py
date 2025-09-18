import json
import os
from logging import getLogger

from dotenv import load_dotenv
from openai import OpenAI

from utils.logger import LogLevelEmoji

logger = getLogger(__name__)


load_dotenv()


endpoint = os.getenv("OPEN_AI_END_POINT")
deployment_name = os.getenv("OPEN_AI_DEPLOYMENT_NAME")
api_key = os.getenv("OPEN_AI_KEY")  # use the api_key


class llmAskResponse:
    def __init__(self, task_done: bool, command: str, result: str | None):
        self.task_done: bool = task_done
        self.command: str = command
        self.result: str = result


class OpenAIApi:

    def __init__(self, system_prompt, deployment_name=deployment_name):
        self.ai_client = OpenAI(base_url=f"{endpoint}", api_key=api_key)
        self.deployment_name = deployment_name
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]

    def ask(self, message) -> llmAskResponse:
        self.messages.append({"role": "user", "content": message})
        return_value = llmAskResponse(False, "", None)
        while self._validate_llm_response(return_value) is False:
            response = self.ai_client.responses.create(
                model=self.deployment_name,
                input=self.messages,
            )
            try:
                return_value = json.loads(response.output_text)
                return_value = llmAskResponse(
                    task_done=return_value.get("task_done", False),
                    command=return_value.get("command", ""),
                    result=return_value.get("result", None),
                )
            except Exception as e:
                logger.error(
                    f"%s Error occurred while dumping JSON: {e}", LogLevelEmoji.ERROR
                )
                logger.error(
                    "%s Failed to parse JSON from LLM response and the response is",
                    LogLevelEmoji.ERROR,
                )
                logger.error(response.output_text)

        self.messages.append({"role": "assistant", "content": response.output_text})
        return return_value

    def clear_history(self):
        self.messages = [
            {
                "role": "user",
                "content": self.system_prompt,
            }
        ]
        return True

    def _validate_llm_response(self, response: llmAskResponse) -> bool:

        if (
            response.task_done is not None
            and response.command is not None
            and response.result is not None
        ):
            logger.info(
                "%s Validated LLM response: %s",
                LogLevelEmoji.INFO,
                json.dumps(response.__dict__),
            )
            return True
        return False
