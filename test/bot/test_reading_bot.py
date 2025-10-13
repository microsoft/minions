"""
This test uses the ReadingBot to analyze the problem statement
https://github.com/SWE-agent/test-repo/blob/main/problem_statements/22.md
and get suggestion for fixing it.
"""

import os
import sys

import pytest

# Add src directory to path to import from local source
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

import logging
logging.basicConfig(level=logging.INFO)

from microbots import ReadingBot
from microbots.MicroBot import BotRunResult

@pytest.mark.integration
def test_reading_bot(test_repo, issue_22):
    issue_text = issue_22[0]

    readingBot = ReadingBot(
        model="azure-openai/mini-swe-agent-gpt5",
        folder_to_mount=str(test_repo)
    )

    response: BotRunResult = readingBot.run(
        issue_text, timeout_in_seconds=300
    )

    print(f"Status: {response.status}, Result: {response.result}, Error: {response.error}")

    assert response.status
    assert response.result is not None