import logging
import os

from agent import Minion

# Configure the basic logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


print("creating reading agent")
readingAgent = Minion(
    agent_type="READING_AGENT",
    model="openai/mini-swe-agent-gpt5",
    # convert into absolute path for folder to mount through os.path.abspath
    folder_to_mount=os.path.abspath("code"),
)

print("starting reading agent")
result = readingAgent.run(
    "read all the countries inside countries.txt and summarize them which is inside code directory, Give me the result countries as Array of strings"
)
print("The result is ----------------------------------")
print(result)
