###############################################################################
################### Ollama Local LLM Interface Setup ##########################
###############################################################################
#
# Install Ollama from https://ollama.com/
# ```
# curl -fsSL https://ollama.com/install.sh | sh
# ollama --version
# ```
#
# Pull and run a local model (e.g., codellama:latest)
# ```
# ollama pull codellama:latest
# ollama serve codellama:latest --port 11434
# ```
#
# Set environment variables in a .env file or your system environment:
# ```
# LOCAL_MODEL_NAME=codellama:latest
# LOCAL_MODEL_PORT=11434
# ```
#
# To use with Microbot, define you Microbot as following
# ```python
# bot = Microbot(
#   model="ollama-local/codellama:latest",
#   folder_to_mount=str(test_repo)
#   )
# ```
###############################################################################

import json
import os
from dataclasses import asdict

from dotenv import load_dotenv
from microbots.llm.llm import LLMAskResponse, LLMInterface
import requests
import logging

logger = logging.getLogger(__name__)

load_dotenv()

LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME") or None
LOCAL_MODEL_PORT = os.getenv("LOCAL_MODEL_PORT") or None

class OllamaLocal(LLMInterface):
    def __init__(self, system_prompt, model_name=LOCAL_MODEL_NAME, model_port=LOCAL_MODEL_PORT, max_retries=3):
        self.model_name = model_name
        self.model_port = model_port
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]

        if not self.model_name or not self.model_port:
            raise ValueError("LOCAL_MODEL_NAME and LOCAL_MODEL_PORT environment variables must be set for or passed as arguments OllamaLocal.")

        # Set these values here. This logic will be handled in the parent class.
        self.max_retries = max_retries
        self.retries = 0

    def ask(self, message) -> LLMAskResponse:
        self.retries = 0 # reset retries for each ask. Handled in parent class.

        self.messages.append({"role": "user", "content": message})

        valid = False
        while not valid:
            response = self._send_request_to_local_model(self.messages)
            valid, askResponse = self._validate_llm_response(response=response)

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

    def _send_request_to_local_model(self, messages):
        logger.debug(f"Sending request to local model {self.model_name} at port {self.model_port}")
        logger.debug(f"Messages: {messages}")
        server = f"http://localhost:{self.model_port}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": json.dumps(messages),
            "stream": False
        }
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(server, json=payload, headers=headers)
        logger.debug(f"\nResponse Code: {response.status_code}\nResponse Text:\n{response.text}\n---")
        if response.status_code == 200:
            response_json = response.json()
            logger.debug(f"\nResponse JSON: {response_json}")
            response_back = response_json.get("response", {})

            # However, as instructed, Ollama is not providing the response only in JSON.
            # It adds some extra text above or below the json sometimes.
            # So, this hack to extract the json part from the response.
            try:
                response_back = response_back.split("{", 1)[1]
                response_back = "{" + response_back.rsplit("}", 1)[0] + "}"
            except Exception as e:
                logger.error(f"Error while extracting JSON from response: {e}")
                raise e

            logger.debug(f"\nResponse from local model: {response_back}")
            return response_back
        else:
            raise Exception(f"Error from local model server: {response.status_code} - {response.text}")
