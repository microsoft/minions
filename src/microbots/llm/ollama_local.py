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
# Pull and run a local model (e.g., qwen3-coder:latest)
# ```
# ollama pull qwen3-coder:latest
# ollama serve qwen3-coder:latest --port 11434
# ```
#
# Set environment variables in a .env file or your system environment:
# ```
# LOCAL_MODEL_NAME=qwen3-coder:latest
# LOCAL_MODEL_PORT=11434
# ```
#
# To use with Microbot, define your Microbot as following
# ```python
# bot = Microbot(
#   model="ollama-local/qwen3-coder:latest",
#   folder_to_mount=str(test_repo)
#   )
# ```
###############################################################################

import json
import os
from dataclasses import asdict

from dotenv import load_dotenv
from microbots.llm.llm import LLMAskResponse, LLMInterface, llm_output_format_str
import requests
import logging

logger = logging.getLogger(__name__)

load_dotenv()

class OllamaLocal(LLMInterface):
    def __init__(self, system_prompt, model_name=None, model_port=None, max_retries=3):
        self.model_name = model_name or os.environ.get("LOCAL_MODEL_NAME")
        self.model_port = model_port or os.environ.get("LOCAL_MODEL_PORT")
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]

        if not self.model_name or not self.model_port:
            raise ValueError("LOCAL_MODEL_NAME and LOCAL_MODEL_PORT environment variables must be set or passed as arguments to OllamaLocal.")

        # Set these values here. This logic will be handled in the parent class.
        self.max_retries = max_retries
        self.retries = 0

    def ask(self, message) -> LLMAskResponse:
        self.retries = 0  # reset retries for each ask. Handled in parent class.

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
        # Set timeout: 30 seconds connect, 600 seconds read to handle model cold start
        response = requests.post(server, json=payload, headers=headers, timeout=(30, 120))
        logger.debug(f"\nResponse Code: {response.status_code}\nResponse Text:\n{response.text}\n---")
        if response.status_code == 200:
            response_json = response.json()
            logger.debug(f"\nResponse JSON: {response_json}")
            return response_json.get("response", "")
        else:
            raise Exception(f"Error from local model server: {response.status_code} - {response.text}")

    def _validate_llm_response(self, response):
        # However, as instructed, Ollama is not providing the response only in JSON.
        # It adds some extra text above or below the JSON sometimes.
        # So, this hack extracts the JSON part from the response.
        try:
            response = response.split("{", 1)[1]
            response = "{" + response.rsplit("}", 1)[0] + "}"
        except Exception as e:
            self.retries += 1
            logger.warning("No JSON in LLM response.\nException: %s\nRetrying... (%d/%d)", e, self.retries, self.max_retries)
            self.messages.append({"role": "user", "content": "LLM_RES_ERROR: Please respond in the following JSON format.\n" + llm_output_format_str})
            return False, None

        logger.debug(f"\nResponse from local model: {response}")
        return super()._validate_llm_response(response)
