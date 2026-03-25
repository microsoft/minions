"""
Integration test for CopilotApi — end-to-end code fix using GitHub Copilot.

This test uses the WritingBot with the `github-copilot` provider to fix
a real syntax error (missing colon) from the SWE-agent test repository.

Prerequisites:
  - GitHub Copilot CLI installed and in PATH (`copilot --version`)
  - Authenticated via `copilot` login or GITHUB_TOKEN / GH_TOKEN env var
  - Active GitHub Copilot subscription
  - Install the ghcp extra: `pip install microbots[ghcp]`
  - Docker daemon running

Usage:
------
  # Run the integration test:
  pytest test/bot/test_writing_bot_copilot.py -v -m "integration"
"""

import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

import logging
logging.basicConfig(level=logging.INFO)

from microbots import WritingBot, BotRunResult


def _copilot_cli_available():
    """Check if the Copilot CLI is installed and accessible."""
    return shutil.which("copilot") is not None


def _copilot_sdk_installed():
    """Check if the github-copilot-sdk package is installed."""
    try:
        import copilot  # noqa: F401
        return True
    except ImportError:
        return False


def _copilot_auth_available():
    """Check if GitHub authentication is available for Copilot."""
    if os.environ.get("GITHUB_TOKEN"):
        return True
    # Check if gh CLI is authenticated
    if shutil.which("gh"):
        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True, timeout=10,
            )
            return result.returncode == 0
        except Exception:
            pass
    return False


skip_no_copilot_cli = pytest.mark.skipif(
    not _copilot_cli_available(),
    reason="GitHub Copilot CLI not installed (copilot not in PATH)",
)

skip_no_copilot_sdk = pytest.mark.skipif(
    not _copilot_sdk_installed(),
    reason="github-copilot-sdk not installed (pip install microbots[ghcp])",
)

skip_no_copilot_auth = pytest.mark.skipif(
    not _copilot_auth_available(),
    reason="No GitHub auth available (set GITHUB_TOKEN or run 'gh auth login')",
)


@skip_no_copilot_cli
@skip_no_copilot_sdk
@skip_no_copilot_auth
@pytest.mark.integration
@pytest.mark.slow
def test_writing_bot_copilot_fixes_syntax_error(test_repo, issue_1):
    """
    End-to-end test: WritingBot with GitHub Copilot fixes a syntax error.

    The test-repo contains `tests/missing_colon.py` with a SyntaxError
    (missing colon on a function definition). The WritingBot should:
    1. Read the error description
    2. Find the faulty file
    3. Fix the syntax error (add the missing colon)
    4. Verify the fix by running the script

    After the bot completes, `verify_function` confirms the fix by
    executing the script and asserting returncode == 0.
    """
    issue_text = issue_1[0]
    verify_function = issue_1[1]

    model = "github-copilot/gpt-4.1"

    writing_bot = WritingBot(
        model=model,
        folder_to_mount=str(test_repo),
    )

    response: BotRunResult = writing_bot.run(
        issue_text,
        max_iterations=25,
        timeout_in_seconds=300,
    )

    print(
        f"Status: {response.status}, "
        f"Result: {response.result}, "
        f"Error: {response.error}"
    )

    assert response.status is True, (
        f"WritingBot did not complete the task. Error: {response.error}"
    )

    # Verify the fix actually works: run the script, expect exit code 0
    verify_function(test_repo)
