import os
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from agent import AgentRunResult
from agents.ReadingAgent import ReadingAgent

myAgent = ReadingAgent(
    model="openai/mini-swe-agent-gpt5",
    folder_to_mount=str(Path(__file__).parent / "countries_dir"),
)

response: AgentRunResult = myAgent.run(
    "Read the /workdir/countries_dir/countries.txt give me the capitals of each country.",
    timeout_in_seconds=300,
)

print(
    f"Status: {response.status}\n***Result:***\n{response.result}\n===\nError: {response.error}"
)
