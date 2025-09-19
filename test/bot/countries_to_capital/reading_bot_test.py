import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Add src directory to path to import from local source
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)
from microbots.bot.ReadingBot import ReadingBot
from microbots.constants import DOCKER_WORKING_DIR
from microbots.MicroBot import BotRunResult

myBot = ReadingBot(
    model="azure-openai/mini-swe-agent-gpt5",
    folder_to_mount=str(Path(__file__).parent / "countries_dir"),
)

response: BotRunResult = myBot.run(
    f"Read the /{DOCKER_WORKING_DIR}/countries_dir/countries.txt give me the capitals of each country.",
    timeout_in_seconds=300,
)

print(
    f"Status: {response.status}\n***Result:***\n{response.result}\n===\nError: {response.error}"
)
