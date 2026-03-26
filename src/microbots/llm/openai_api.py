import json
import os
from dataclasses import asdict

from dotenv import load_dotenv
from openai import OpenAI
from microbots.llm.llm import LLMAskResponse, LLMInterface
from microbots.llm.token_provider import TokenProvider, env_token_provider

load_dotenv()

endpoint = os.getenv("OPEN_AI_END_POINT")
deployment_name = os.getenv("OPEN_AI_DEPLOYMENT_NAME")


class OpenAIApi(LLMInterface):

    def __init__(self, system_prompt, deployment_name=deployment_name, max_retries=3, token_provider: TokenProvider | None = None):
        self.token_provider = token_provider or env_token_provider("OPEN_AI_KEY")
        self.ai_client = OpenAI(base_url=f"{endpoint}", api_key=self.token_provider())
        self.deployment_name = deployment_name
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]

        # Set these values here. This logic will be handled in the parent class.
        self.max_retries = max_retries
        self.retries = 0

    def ask(self, message) -> LLMAskResponse:
        self.retries = 0 # reset retries for each ask. Handled in parent class.
        self.ai_client.api_key = self.token_provider()

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

