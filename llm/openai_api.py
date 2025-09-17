import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

# sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# from agent_utils import llm_output_format

endpoint = os.getenv("OPEN_AI_END_POINT")
deployment_name = os.getenv("OPEN_AI_DEPLOYMENT_NAME")
api_key = os.getenv("OPEN_AI_KEY")  # use the api_key


class llmAskResponse:
    def __init__(self):
        self.task_done: bool = False
        self.command: str = ""
        self.result: str | None = None


def validate_llm_response(response: dict) -> bool:
    if "task_done" in response and "command" in response and "result" in response:
        return True
    return False


class OpenAIApi:

    def __init__(self, system_prompt, deployment_name=deployment_name):
        self.ai_client = OpenAI(base_url=f"{endpoint}", api_key=api_key)
        self.deployment_name = deployment_name
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]

    def ask(self, message) -> dict:
        self.messages.append({"role": "user", "content": message})
        return_value = {}
        while validate_llm_response(return_value) is False:
            response = self.ai_client.responses.create(
                model=self.deployment_name,
                input=self.messages,
            )
            try:
                return_value = json.loads(response.output_text)
            except Exception as e:
                print(f"Error occurred while dumping JSON: {e}")
                print("Failed to parse JSON from LLM response and the response is")
                print(response.output_text)

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


# Do a quick test
# if __name__ == "__main__":
#     openai_api = OpenAIApi(
#         system_prompt="""You are a helpful assistant.
#         I will give you a task to achieve using the shell.
#         "You will provide the result of the task in this particular below json format
#         {llm_output_format}
#         if the task is completed change task_done to true
#         """
#     )
#     response = openai_api.ask(
#         "The task is give me the command to list all files in current directory"
#     )
#     print(response)
