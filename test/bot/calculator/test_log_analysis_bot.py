"""
Integration tests for LogAnalysisBot
"""
import pytest
import logging
import os
import sys
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# Add src directory to path to import from local source
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)

from microbots.bot.LogAnalysisBot import LogAnalysisBot
from microbots.constants import DOCKER_WORKING_DIR, LOG_FILE_DIR
from microbots.MicroBot import BotRunResult


@pytest.mark.integration
@pytest.mark.docker
@pytest.mark.slow
class TestLogAnalysisBotIntegration:
    """Integration tests for LogAnalysisBot """

    @pytest.fixture(scope="class")
    def code_dir(self):
        """Get the path to the calculator code directory"""
        return Path(__file__).parent / "code"

    @pytest.fixture(scope="class")
    def log_file(self):
        """Get the path to the calculator log file"""
        return Path(__file__).parent / "calculator.log"

    @pytest.fixture(scope="class")
    def log_analysis_bot(self, code_dir):
        """Create a LogAnalysisBot instance """
        # Ensure the code directory exists
        if not code_dir.exists():
            pytest.skip(f"Code directory not found: {code_dir}")
        
        bot = LogAnalysisBot(
            model="azure-openai/mini-swe-agent-gpt5",
            folder_to_mount=str(code_dir),
        )
        yield bot
        
        # Cleanup: stop the environment
        if hasattr(bot, 'environment') and bot.environment:
            try:
                bot.environment.stop()
            except Exception as e:
                logger.warning(f"Error stopping environment: {e}")

    def test_analyze_calculator_log(self, log_analysis_bot, log_file):
        """Test LogAnalysisBot analyzing calculator log file"""
        # Ensure the log file exists
        if not log_file.exists():
            pytest.skip(f"Log file not found: {log_file}")

        # Run the bot with the log analysis task
        response: BotRunResult = log_analysis_bot.run(
            str(log_file),
            timeout_in_seconds=300,
        )
        
        # Log the response for debugging
        logger.info(f"Status: {response.status}")
        logger.info(f"Result: {response.result}")
        if response.error:
            logger.error(f"Error: {response.error}")
        
        # Assertions
        assert response.status == True, f"Bot run failed with error: {response.error}"
        assert response.result is not None, "Bot result should not be None"
        assert response.error is None, f"Bot should not have errors! ERROR: {response.error}"
        
        # Check that the result contains analysis of the log
        result_lower = response.result.lower()
        assert len(response.result.strip()) > 0, "Result should not be empty"

    def test_analyze_nonexistent_log(self, log_analysis_bot):
        """Test LogAnalysisBot behavior when trying to analyze a non-existent log file"""
        nonexistent_log = Path(__file__).parent / "nonexistent.log"
        
        # LogAnalysisBot should raise a ValueError when trying to copy a nonexistent file
        # This is expected behavior at the infrastructure level
        with pytest.raises(ValueError, match="Failed to copy file to container"):
            response: BotRunResult = log_analysis_bot.run(
                str(nonexistent_log),
                timeout_in_seconds=60,
            )
            
        logger.info(f"Successfully caught expected ValueError for nonexistent log file")

    

# Manual test runner function (can be called directly)
def run_log_analysis_bot_manual_test():
    """Manual test function that can be run outside pytest"""
    print("=== Manual LogAnalysisBot Integration Test ===")

    code_dir = Path(__file__).parent / "code"
    log_file = Path(__file__).parent / "calculator.log"

    if not code_dir.exists():
        print(f"ERROR: Code directory not found: {code_dir}")
        return
    
    if not log_file.exists():
        print(f"ERROR: Log file not found: {log_file}")
        return
    
    print(f"Using code from: {code_dir}")
    print(f"Using log file: {log_file}")
    
    try:
        # Create LogAnalysisBot instance
        myBot = LogAnalysisBot(
            model="azure-openai/mini-swe-agent-gpt5",
            folder_to_mount=str(code_dir),
        )
        
        # Run the log analysis task
        response: BotRunResult = myBot.run(
            str(log_file),
            timeout_in_seconds=300,
        )
        
        # Print results
        print(f"Status: {response.status}")
        print(f"***Result:***\n{response.result}")
        print(f"===\nError: {response.error}")
        
        # Cleanup
        if hasattr(myBot, 'environment') and myBot.environment:
            myBot.environment.stop()
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Allow running this file directly for manual testing
    run_log_analysis_bot_manual_test()
