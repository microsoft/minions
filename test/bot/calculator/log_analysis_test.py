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

from microbots.bot.LogAnalysisBot import LogAnalysisBot
from microbots.constants import DOCKER_WORKING_DIR, LOG_FILE_DIR
from microbots.MicroBot import BotRunResult

myBot = LogAnalysisBot(
    model="azure-openai/mini-swe-agent-gpt5",
    folder_to_mount=str(Path(__file__).parent / "code"),
)

response: BotRunResult = myBot.run(
    str(Path(__file__).parent / "calculator.log"),
    timeout_in_seconds=300,
)

print(
    f"Status: {response.status}\n***Result:***\n{response.result}\n===\nError: {response.error}"
)
