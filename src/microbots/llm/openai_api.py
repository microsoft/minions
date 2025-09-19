import json
import os
from dataclasses import dataclass
from logging import getLogger

from dotenv import load_dotenv
from openai import OpenAI
from microbots.utils.logger import LogLevelEmoji

logger = getLogger(__name__)


load_dotenv()

from openai import OpenAI

endpoint = os.getenv("OPEN_AI_END_POINT")
deployment_name = os.getenv("OPEN_AI_DEPLOYMENT_NAME")
api_key = os.getenv("OPEN_AI_KEY")  # use the api_key


@dataclass
class llmAskResponse:
    task_done: bool = False
    command: str = ""
    result: str | None = None


class OpenAIApi:

    def __init__(self, system_prompt, deployment_name=deployment_name):
        self.ai_client = OpenAI(base_url=f"{endpoint}", api_key=api_key)
        self.deployment_name = deployment_name
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]

    def ask(self, message) -> llmAskResponse:
        self.messages.append({"role": "user", "content": message})
        return_value = {}
        while self._validate_llm_response(return_value) is False:
            response = self.ai_client.responses.create(
                model=self.deployment_name,
                input=self.messages,
            )
            try:
                return_value = json.loads(response.output_text)
            except Exception as e:
                logger.error(
                    f"%s Error occurred while dumping JSON: {e}", LogLevelEmoji.ERROR
                )
                logger.error(
                    "%s Failed to parse JSON from LLM response and the response is",
                    LogLevelEmoji.ERROR,
                )
                logger.error(response.output_text)

        self.messages.append({"role": "assistant", "content": json.dumps(return_value)})

        return llmAskResponse(
            task_done=return_value["task_done"],
            result=return_value["result"],
            command=return_value["command"],
        )

    def clear_history(self):
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]
        return True

    def _validate_llm_response(self, response: dict) -> bool:
        if "task_done" in response and "command" in response and "result" in response:
            logger.info("The llm response is %s ", response)
            return True
        return False
