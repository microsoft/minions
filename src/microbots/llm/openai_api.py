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
        self.ai_client = OpenAI(base_url=f"{endpoint}", api_key=api_key)
        self.deployment_name = deployment_name
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]

        # Set these values here. This logic will be handled in the parent class.
        self.max_retries = max_retries
        self.retries = 0

    def ask(self, message) -> LLMAskResponse:
        self.retries = 0 # reset retries for each ask. Handled in parent class.

        self.messages.append({"role": "user", "content": message})

        valid = False
        while not valid:
            response = self.ai_client.responses.create(
                model=self.deployment_name,
                input=self.messages,
            )
            valid, askResponse = self._validate_llm_response(response=response.output_text)

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

