"""
Integration test for LeadBot.

LeadBot decomposes a task into subtasks and delegates each one to a sub_agent
running inside the Docker sandbox.  This test uses the same SWE-agent test-repo
issue #1 (missing-colon syntax error) that the WritingBot test uses, but goes
through the LeadBot → sub_agent → WritingBot delegation chain.

Usage:
------
# Run the Azure OpenAI integration test:
pytest test/bot/test_lead_bot.py::test_lead_bot_azure -v

# Run all LeadBot tests:
pytest test/bot/test_lead_bot.py -v
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

from microbots import LeadBot, BotRunResult


@pytest.mark.integration
@pytest.mark.slow
def test_lead_bot_azure(test_repo, issue_1):
    """Test LeadBot delegates to sub_agent and solves a simple syntax-fix task.

    The test verifies the full orchestration chain:
        LeadBot (LLM) → sub_agent CLI (inside Docker) → WritingBot → fix applied

    The issue is a missing colon in ``tests/missing_colon.py``, the same one
    used by the WritingBot integration test.
    """
    os.setenv("AZURE_OPENAI_LLM", "azure-openai/" + os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "mini-swe-agent-gpt5"))
    issue_text = issue_1[0]
    verify_function = issue_1[1]
    model = f"azure-openai/{os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'mini-swe-agent-gpt5')}"

    lead_bot = LeadBot(
        model=model,
        folder_to_mount=str(test_repo),
    )

    response: BotRunResult = lead_bot.run(
        task=issue_text,
        max_iterations=50,
        timeout_in_seconds=600,
    )

    print(
        f"Status: {response.status}, "
        f"Result: {response.result}, "
        f"Error: {response.error}"
    )

    # The lead bot should complete successfully
    assert response.status, f"LeadBot failed with error: {response.error}"
    assert response.result is not None
    assert response.error is None

    # The actual code fix should have been applied by the sub_agent
    verify_function(test_repo)
