import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from agent import Agent
from agent_utils import AgentRunResult

myAgent = Agent(
    agent_type = "READING_AGENT",
    model = "openai/mini-swe-agent-gpt5",
    folder_to_mount = str(Path(__file__).parent / "countries_dir")
)

response : AgentRunResult = myAgent.run("Read the /countries_dir/countries.txt file and return me the capital of each countries in the list as a json object")

print(
    f"Status: {response.status}, Result: {response.result}, Error: {response.error}"
)