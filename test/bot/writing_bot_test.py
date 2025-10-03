"""
This test uses the WritingBot to solve https://github.com/SWE-agent/test-repo/issues/1
The issue is a simple syntax correction issue from original SWE-bench's test-repo.
"""

import os
import sys
# Add src directory to path to import from local source
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

import logging
logging.basicConfig(level=logging.INFO)

from microbots import WritingBot
from microbots.MicroBot import BotRunResult

def test_writing_bot(test_repo, issue_1):
    issue_text = issue_1[0]
    verify_function = issue_1[1]

    writingBot = WritingBot(
        model="azure-openai/mini-swe-agent-gpt5",
        folder_to_mount=str(test_repo)
    )

    response: BotRunResult = writingBot.run(
        issue_text, timeout_in_seconds=300
    )

    print(f"Status: {response.status}, Result: {response.result}, Error: {response.error}")

    verify_function(test_repo)