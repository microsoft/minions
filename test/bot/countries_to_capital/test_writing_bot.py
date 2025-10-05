"""
Integration tests for WritingBot
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

from microbots.bot.WritingBot import WritingBot
from microbots.constants import DOCKER_WORKING_DIR
from microbots.MicroBot import BotRunResult


@pytest.mark.integration
@pytest.mark.docker
@pytest.mark.slow
class TestWritingBotIntegration:
    """Integration tests for WritingBot with real environment and API"""

    @pytest.fixture(scope="class")
    def countries_data_dir(self):
        """Get the path to the countries test data directory"""
        return Path(__file__).parent / "countries_dir"

    @pytest.fixture(scope="class")
    def writing_bot(self, countries_data_dir):
        """Create a WritingBot instance with real environment"""
        # Ensure the countries data directory exists
        if not countries_data_dir.exists():
            pytest.skip(f"Countries data directory not found: {countries_data_dir}")
        
        bot = WritingBot(
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

    def test_write_capitals_file(self, writing_bot, countries_data_dir):
        """Test WritingBot reading countries.txt and creating capitals.txt file"""
        # Clean up any existing capitals.txt file
        capitals_file = countries_data_dir / "capitals.txt"
        if capitals_file.exists():
            capitals_file.unlink()
        
        # Run the bot with the writing task
        response: BotRunResult = writing_bot.run(
            f"Read the /{DOCKER_WORKING_DIR}/countries_dir/countries.txt and store their capitals in /{DOCKER_WORKING_DIR}/countries_dir/capitals.txt file",
            timeout_in_seconds=300,
        )
        
        # Log the response for debugging
        logger.info(f"Status: {response.status}")
        logger.info(f"Result: {response.result}")
        if response.error:
            logger.error(f"Error: {response.error}")
        
        # Basic assertions
        assert response.status == True, f"Bot run failed with error: {response.error}"
        assert response.result is not None, "Bot result should not be None"
        assert response.error is None, f"Bot should not have errors: {response.error}"
        
        # Check that the capitals.txt file was created
        assert capitals_file.exists(), f"Bot should have created capitals.txt file at {capitals_file}"
        
        # Check the content of the created file
        capitals_content = capitals_file.read_text().strip()
        assert len(capitals_content) > 0, "capitals.txt should not be empty"
        logger.info(f"Created capitals.txt content: {capitals_content}")



# Manual test runner function (can be called directly)
def run_writing_bot_manual_test():
    """Manual test function that can be run outside pytest"""
    print("=== Manual WritingBot Integration Test ===")

    countries_data_dir = Path(__file__).parent / "countries_dir"

    if not countries_data_dir.exists():
        print(f"ERROR: Countries data directory not found: {countries_data_dir}")
        return
    
    print(f"Using countries data from: {countries_data_dir}")
    
    try:
        # Create WritingBot instance
        myBot = WritingBot(
            model="azure-openai/mini-swe-agent-gpt5",
            folder_to_mount=str(countries_data_dir),
        )
        
        # Run the writing task
        response: BotRunResult = myBot.run(
            f"Read the /{DOCKER_WORKING_DIR}/countries_dir/countries.txt and store their capitals in /{DOCKER_WORKING_DIR}/countries_dir/capitals.txt file",
            timeout_in_seconds=300,
        )
        
        # Print results
        print(f"Status: {response.status}")
        print(f"***Result:***\n{response.result}")
        print(f"===\nError: {response.error}")
        
        # Check if file was created
        capitals_file = countries_data_dir / "capitals.txt"
        if capitals_file.exists():
            print(f"✅ capitals.txt was created!")
            print(f"Content: {capitals_file.read_text()}")
        else:
            print("❌ capitals.txt was not created")
        
        # Cleanup
        if hasattr(myBot, 'environment') and myBot.environment:
            myBot.environment.stop()
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Allow running this file directly for manual testing
    run_writing_bot_manual_test()
