"""
This test uses the LogAnalysisBot to analyze logs from a failing run mentioned in
https://github.com/SWE-agent/test-repo/issues/1
The issue is a simple syntax correction issue from original SWE-bench's test-repo.
"""

import os
import subprocess
import sys

import pytest
# Add src directory to path to import from local source
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from microbots import LogAnalysisBot, BotRunResult

@pytest.mark.integration
@pytest.mark.docker

class TestLogAnalysisBot:

    @pytest.fixture(scope="function")
    def log_file_path(self, tmpdir):
        assert tmpdir.exists()
        yield tmpdir / "error.log"
        if tmpdir.exists():
            subprocess.run(["rm", "-rf", str(tmpdir)])

    @pytest.fixture(scope="function")
    def log_analysis_bot(self, test_repo):
        model = f"azure-openai/{os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'mini-swe-agent-gpt5')}"
        log_analysis_bot = LogAnalysisBot(
            model=model,
            folder_to_mount=str(test_repo)
        )

        yield log_analysis_bot

        # Cleanup: stop the environment
        if hasattr(log_analysis_bot, 'environment') and log_analysis_bot.environment:
            try:
                log_analysis_bot.environment.stop()
            except Exception as e:
                logger.warning(f"Error stopping environment: {e}")

    @pytest.mark.slow
    def test_log_analysis_bot(self, log_analysis_bot, log_file_path, test_repo, issue_1):
        assert log_analysis_bot is not None

        run_function = issue_1[2]

        try:
            result = run_function(test_repo)
        except Exception as e:
            pytest.fail(f"Failed to run function to generate logs: {e}")

        assert result.returncode != 0
        assert result.stderr is not None

        with open(log_file_path, "w") as log_file:
            log_file.write(result.stderr)

        response: BotRunResult = log_analysis_bot.run(
            str(log_file_path), timeout_in_seconds=300
        )

        print(f"Status: {response.status}, Result: {response.result}, Error: {response.error}")

        assert response.status
        assert response.result is not None
        assert response.error is None

    def test_log_analysis_bot_nonexistent_file(self, log_analysis_bot):
        assert log_analysis_bot is not None

        fake_log_file = "non_existent.log"

        with pytest.raises(ValueError, match="Failed to copy additional mount to container"):
            log_analysis_bot.run(
                fake_log_file, timeout_in_seconds=60
            )

        logger.info("Successfully caught expected ValueError for nonexistent log file")

    @pytest.mark.slow
    def test_log_analysis_bot_max_iterations(self, log_analysis_bot, log_file_path, test_repo, issue_1):
        """Test that max_iterations parameter limits the number of iterations"""
        assert log_analysis_bot is not None

        run_function = issue_1[2]

        try:
            result = run_function(test_repo)
        except Exception as e:
            pytest.fail(f"Failed to run function to generate logs: {e}")

        assert result.returncode != 0
        assert result.stderr is not None

        with open(log_file_path, "w") as log_file:
            log_file.write(result.stderr)

        # Run with a very low max_iterations to force it to hit the limit
        response: BotRunResult = log_analysis_bot.run(
            str(log_file_path), max_iterations=2, timeout_in_seconds=300
        )

        print(f"Status: {response.status}, Result: {response.result}, Error: {response.error}")

        # Should fail due to max iterations being reached
        assert response.status is False
        assert response.error is not None
        assert "Max iterations 2 reached" in response.error

        logger.info("Successfully verified max_iterations parameter limits execution")
