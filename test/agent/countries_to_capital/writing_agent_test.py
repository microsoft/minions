import os
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from agent import AgentRunResult, Minion

myAgent = Minion(
    agent_type = "WRITING_AGENT",
    model = "openai/mini-swe-agent-gpt5",
    folder_to_mount = str(Path(__file__).parent / "countries_dir")
)

response : AgentRunResult = myAgent.run("Read the /app/countries_dir/countries.txt store their capitals in /app/countries_dir/capitals.txt file", timeout_in_seconds=300)

print(
    f"Status: {response.status}, Result: {response.result}, Error: {response.error}"
)