import json
import os
from dataclasses import asdict

from dotenv import load_dotenv
from openai import OpenAI
from microbots.llm.llm import LLMAskResponse, LLMInterface

load_dotenv()

endpoint = os.getenv("OPEN_AI_END_POINT")
deployment_name = os.getenv("OPEN_AI_DEPLOYMENT_NAME")
api_key = os.getenv("OPEN_AI_KEY")  # use the api_key


class OpenAIApi(LLMInterface):

    def __init__(self, system_prompt, deployment_name=deployment_name, max_retries=3):
        self.deployment_name = deployment_name
        self.ai_client = OpenAI(base_url=f"{endpoint}", api_key=api_key)
        super().__init__(system_prompt=system_prompt, max_retries=max_retries)

    def ask(self, message) -> LLMAskResponse:
        self.retries = 0 # reset retries for each ask. Handled in parent class.

        self.messages.append({"role": "user", "content": message})

        valid = False
        while not valid:
            response = self.ai_client.responses.create(
                model=self.deployment_name,
                input=self.messages,
            )
            self.messages.append({"role": "assistant", "content": response.output_text})
            valid, askResponse = self._validate_llm_response(response=response.output_text)

        # Remove last assistant message and replace with structured response
        self.messages.pop()
        self.messages.append({"role": "assistant", "content": json.dumps(asdict(askResponse))})

        return askResponse

    def clear_history(self):
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]
        return True

