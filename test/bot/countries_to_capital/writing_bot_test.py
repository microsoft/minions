import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from bot.WritingBot import WritingBot
from MicroBot import BotRunResult

myBot = WritingBot(
    model="openai/mini-swe-agent-gpt5",
    folder_to_mount=str(Path(__file__).parent / "countries_dir"),
)

response: BotRunResult = myBot.run(
    "Read the /workdir/countries_dir/countries.txt store their capitals in /workdir/countries_dir/capitals.txt file",
    timeout_in_seconds=300,
)

print(f"Status: {response.status}, Result: {response.result}, Error: {response.error}")
