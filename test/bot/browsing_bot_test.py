import logging
import os
import sys


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/")))
from microbots.bot.BrowsingBot import BrowsingBot
from microbots.MicroBot import BotRunResult

myBot = BrowsingBot(
    model="azure-openai/mini-swe-agent-gpt5",
)

response: BotRunResult = myBot.run(
    "What is the capital of France?",
    timeout_in_seconds=300,
)

final_result = response.result
# logger.info(f"Response: {response}")
logger.debug("Status: %s\n, Error: %s\n\n\n, ***Result:***\n %s\n", response.status, response.error, response.result)

print("Final Result: ", final_result)
