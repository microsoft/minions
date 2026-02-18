import json
import os
import re
from dataclasses import asdict
from logging import getLogger

from dotenv import load_dotenv
from anthropic import Anthropic
from microbots.llm.llm import LLMAskResponse, LLMInterface

logger = getLogger(__name__)

load_dotenv()

endpoint = os.getenv("ANTHROPIC_END_POINT")
deployment_name = os.getenv("ANTHROPIC_DEPLOYMENT_NAME")
api_key = os.getenv("ANTHROPIC_API_KEY")


class AnthropicApi(LLMInterface):

    def __init__(self, system_prompt, deployment_name=deployment_name, max_retries=3):
        self.ai_client = Anthropic(
            api_key=api_key,
            base_url=endpoint
        )
        self.deployment_name = deployment_name
        self.system_prompt = system_prompt
        self.messages = []

        # Set these values here. This logic will be handled in the parent class.
        self.max_retries = max_retries
        self.retries = 0

    def ask(self, message) -> LLMAskResponse:
        self.retries = 0  # reset retries for each ask. Handled in parent class.

        self.messages.append({"role": "user", "content": message})

        valid = False
        while not valid:
            response = self.ai_client.messages.create(
                model=self.deployment_name,
                system=self.system_prompt,
                messages=self.messages,
                max_tokens=4096,
            )

            # Extract text content from response
            response_text = response.content[0].text if response.content else ""
            logger.debug("Raw Anthropic response length: %d chars", len(response_text))
            logger.debug("Raw Anthropic response (first 1000 chars): %s", response_text[:1000])
            
            # Log any control characters found in the response
            control_chars = [f"\\x{ord(c):02x}" for c in response_text if ord(c) < 0x20 and c not in '\n']
            if control_chars:
                logger.debug("Control characters found in response: %s", control_chars[:20])  # Log first 20

            # Try to extract JSON if wrapped in markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                logger.debug("JSON extracted from markdown code block")
                response_text = json_match.group(1)
            else:
                logger.debug("No markdown code block found, using raw response")
            
            logger.debug("Response text to validate (first 500 chars): %s", repr(response_text[:500]))

            valid, askResponse = self._validate_llm_response(response=response_text)

        self.messages.append({"role": "assistant", "content": json.dumps(asdict(askResponse))})

        return askResponse

    def clear_history(self):
        self.messages = []
        return True
