import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_utils import llm_output_format

endpoint = os.getenv("OPEN_AI_END_POINT")
deployment_name = os.getenv("OPEN_AI_DEPLOYMENT_NAME")
api_key = os.getenv("OPEN_AI_KEY")  # use the api_key


class llmAskResponse:
    task_done: bool
    command: str
    result: str | None


class OpenAIResponseApi:

    def __init__(self, system_prompt, deployment_name=deployment_name):
        self.ai_client = OpenAI(base_url=f"{endpoint}", api_key=api_key)
        self.deployment_name = deployment_name
        self.system_prompt = system_prompt
        self.messages = [{"role": "user", "content": system_prompt}]

    def ask(self, message) -> llmAskResponse:
        self.messages.append({"role": "user", "content": message})
        response = self.ai_client.responses.create(
            model=self.deployment_name,
            input=self.messages,
        )
        self.messages.append({"role": "system", "content": response.output_text})
        return_value = {
            "task_done": False,
            "command": None,
            "result": None,
        }
        try:
            return_value = json.loads(response.output_text)
        except Exception as e:
            print(f"Error occurred while dumping JSON: {e}")
            print("Failed to parse JSON from LLM response and the response is")
            print(response.output_text)

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
if __name__ == "__main__":
    openai_api = OpenAIResponseApi(
        system_prompt="""You are a helpful assistant.
        I will give you a task to achieve using the shell.
        "You will provide the result of the task in this particular below json format
        {llm_output_format}
        if the task is completed change task_done to true
        """
    )
    response = openai_api.ask(
        "The task is give me the command to list all files in current directory"
    )
    print(response)
