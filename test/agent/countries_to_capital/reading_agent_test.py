import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from agent import AgentRunResult, Minion

myAgent = Minion(
    agent_type="READING_AGENT",
    model="openai/mini-swe-agent-gpt5",
    folder_to_mount=str(Path(__file__).parent / "countries_dir"),
)

response: AgentRunResult = myAgent.run(
    "Read the /countries_dir/countries.txt give me the capitals of each country.",
    timeout_in_seconds=300,
)

print(
    f"Status: {response.status}\n***Result:***\n{response.result}\n===\nError: {response.error}"
)
