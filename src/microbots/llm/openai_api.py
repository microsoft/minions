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
    reasoning: str | None = None


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
            response = self.ai_client.chat.completions.create(
                model=self.deployment_name,
                messages=self.messages,
            )
            try:
                return_value = json.loads(response.choices[0].message.content)
            except Exception as e:
                logger.error(
                    f"%s Error occurred while dumping JSON: {e}", LogLevelEmoji.ERROR
                )
                logger.error(
                    "%s Failed to parse JSON from LLM response and the response is",
                    LogLevelEmoji.ERROR,
                )
                logger.error(response.choices[0].message.content)

        self.messages.append({"role": "assistant", "content": json.dumps(return_value)})

        return llmAskResponse(
            task_done=return_value["task_done"],
            result=return_value["result"],
            command=return_value["command"],
            reasoning=return_value.get("reasoning", "No reasoning provided"),
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
        required_fields = ["task_done", "command", "result"]
        if all(field in response for field in required_fields):
            logger.info("The llm response is %s ", response)
            return True
        return False
