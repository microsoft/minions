"""
Integration tests for ReadingBot
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

from microbots.bot.ReadingBot import ReadingBot
from microbots.constants import DOCKER_WORKING_DIR
from microbots.MicroBot import BotRunResult


@pytest.mark.integration
@pytest.mark.docker
@pytest.mark.slow
class TestReadingBotIntegration:
    """Integration tests for ReadingBot with real environment and API"""

    @pytest.fixture(scope="class")
    def countries_data_dir(self):
        """Get the path to the countries test data directory"""
        return Path(__file__).parent / "countries_dir"

    @pytest.fixture(scope="class")
    def reading_bot(self, countries_data_dir):
        """Create a ReadingBot instance with real environment"""
        # Ensure the countries data directory exists
        if not countries_data_dir.exists():
            pytest.skip(f"Countries data directory not found: {countries_data_dir}")
        
        bot = ReadingBot(
            model="azure-openai/mini-swe-agent-gpt5",
            folder_to_mount=str(countries_data_dir),
        )
        yield bot
        
        # Cleanup: stop the environment
        if hasattr(bot, 'environment') and bot.environment:
            try:
                bot.environment.stop()
            except Exception as e:
                logger.warning(f"Error stopping environment: {e}")

    def test_read_countries_file(self, reading_bot):
        """Test ReadingBot reading countries.txt file and extracting capitals"""
        # Run the bot with the reading task
        response: BotRunResult = reading_bot.run(
            f"Read the /{DOCKER_WORKING_DIR}/countries_dir/countries.txt file and give me the capitals of each country.",
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
        assert response.error is None, f"Bot should not have errors: {response.error}"
        
        # Check that the result contains actual capitals from your countries
        result_lower = response.result.lower()
        expected_capitals = ["delhi", "washington", "brasília", "berlin", "singapore"]
        assert any(capital in result_lower for capital in expected_capitals), \
            f"Result should contain capitals of the countries in the file. Got: {response.result}"

    def test_read_nonexistent_file(self, reading_bot):
        """Test ReadingBot behavior when trying to read a non-existent file"""
        response: BotRunResult = reading_bot.run(
            f"Read the /{DOCKER_WORKING_DIR}/countries_dir/nonexistent.txt file.",
            timeout_in_seconds=60,
        )
        
        # Log the response for debugging
        logger.info(f"Status: {response.status}")
        logger.info(f"Result: {response.result}")
        if response.error:
            logger.info(f"Error: {response.error}")
        
        # The bot should handle this gracefully - either return an error status
        # or mention in the result that the file doesn't exist
        assert response.status in [True, False], "Bot should handle missing files gracefully"
        
        if response.status == True:
            # If successful, the result should mention the file doesn't exist
            result_lower = response.result.lower()
            assert any(phrase in result_lower for phrase in ["not found", "does not exist", "no such file"]), \
                f"Result should indicate file doesn't exist. Got: {response.result}"


    @pytest.mark.parametrize("task", [
        "Count the number of countries in the countries.txt file",
        "Tell me which country has Paris as its capital",
        "What is the capital of Germany according to the file?",
    ])
    def test_specific_reading_tasks(self, reading_bot, task):
        """Test ReadingBot with specific reading tasks"""
        full_task = f"Read /{DOCKER_WORKING_DIR}/countries_dir/countries.txt and {task.lower()}"
        
        response: BotRunResult = reading_bot.run(
            full_task,
            timeout_in_seconds=120,
        )
        
        # Log the response for debugging
        logger.info(f"Task: {task}")
        logger.info(f"Status: {response.status}")
        logger.info(f"Result: {response.result}")
        if response.error:
            logger.error(f"Error: {response.error}")
        
        # Basic assertions
        assert response.status == True, f"Task '{task}' failed with error: {response.error}"
        assert response.result is not None, f"Task '{task}' result should not be None"
        assert len(response.result.strip()) > 0, f"Task '{task}' should return non-empty result"


# Manual test runner function (can be called directly)
def run_reading_bot_manual_test():
    """Manual test function that can be run outside pytest"""
    print("=== Manual ReadingBot Integration Test ===")

    countries_data_dir = Path(__file__).parent / "countries_dir"

    if not countries_data_dir.exists():
        print(f"ERROR: Countries data directory not found: {countries_data_dir}")
        return
    
    print(f"Using countries data from: {countries_data_dir}")
    
    try:
        # Create ReadingBot instance
        myBot = ReadingBot(
            model="azure-openai/mini-swe-agent-gpt5",
            folder_to_mount=str(countries_data_dir),
        )
        
        # Run the reading task
        response: BotRunResult = myBot.run(
            f"Read the /{DOCKER_WORKING_DIR}/countries_dir/countries.txt file and give me the capitals of each country.",
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
    run_reading_bot_manual_test()