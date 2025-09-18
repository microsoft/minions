import logging
import os
import sys


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from bot.BrowsingBot import BrowsingBot
from MicroBot import BotRunResult

myBot = BrowsingBot(
    model="openai/mini-swe-agent-gpt5",
)

response: BotRunResult = myBot.run(
    "What is the capital of France?",
    timeout_in_seconds=300,
)

logger.info("Status: %s, Result: %s, Error: %s", response.status, response.result, response.error)
