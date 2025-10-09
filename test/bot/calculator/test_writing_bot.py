"""
Integration tests for WritingBot
"""
import pytest
import logging
import os
import sys
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('writing_bot_test.log')
    ]
)
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
    def calculator_data_dir(self):
        """Get the path to the calculator test data directory"""
        return Path(__file__).parent  # The test file is already in the calculator directory

    @pytest.fixture(scope="class")
    def writing_bot(self, calculator_data_dir):
        """Create a WritingBot instance with real environment"""
        # Ensure the calculator data directory exists
        if not calculator_data_dir.exists():
            pytest.skip(f"Calculator data directory not found: {calculator_data_dir}")
        
        # Check if calculator.log exists
        calc_log = calculator_data_dir / "calculator.log"
        if not calc_log.exists():
            pytest.skip(f"Calculator log file not found: {calc_log}")
        
        bot = WritingBot(
            model="azure-openai/mini-swe-agent-gpt5",
            folder_to_mount=str(calculator_data_dir),
        )
        yield bot
        
        # Cleanup: stop the environment
        if hasattr(bot, 'environment') and bot.environment:
            try:
                bot.environment.stop()
            except Exception as e:
                logger.warning(f"Error stopping environment: {e}")

    def test_write_calculator_fix(self, writing_bot, calculator_data_dir):
        """Test WritingBot reading calculator.log and fixing the calculator.py file"""
        # Take a snapshot of git status before running the bot
        import subprocess
        git_before = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, 
                                  cwd=calculator_data_dir.parent.parent.parent)
        
        # Run the bot with the fix task
        response: BotRunResult = writing_bot.run(
            """Inside the mounted directory there is a calculator.log which have execution of code/calculator.py.
               Read the log and fix the error.""",
            timeout_in_seconds=300,
        )
        
        # Log the response for debugging (safely handle potential JSON serialization issues)
        logger.info(f"Status: {response.status}")
      
        # Check if the calculator.py file was modified
        calc_file = calculator_data_dir / "code" / "calculator.py"
        calc_content = calc_file.read_text()
        
        # Check that the fix was applied (should contain some form of zero check)
        assert "b == 0" in calc_content or "b != 0" in calc_content or "if not b" in calc_content or "if b == 0" in calc_content, \
            "Calculator code should contain a check for division by zero"
        
        logger.info(f"Fixed calculator.py content preview: {calc_content[:500]}...")
        


# Manual test runner function (can be called directly)
def run_writing_bot_manual_test():
    """Manual test function that can be run outside pytest"""
    print("=== Manual WritingBot Integration Test ===")

    calculator_data_dir = Path(__file__).parent.parent / "calculator"

    if not calculator_data_dir.exists():
        print(f"ERROR: Calculator data directory not found: {calculator_data_dir}")
        return
    
    calc_log = calculator_data_dir / "calculator.log"
    if not calc_log.exists():
        print(f"ERROR: Calculator log file not found: {calc_log}")
        return
    
    print(f"Using calculator data from: {calculator_data_dir}")
    
    try:
        # Create WritingBot instance
        myBot = WritingBot(
            model="azure-openai/mini-swe-agent-gpt5",
            folder_to_mount=str(calculator_data_dir),
        )
        
        # Run the fix task
        response: BotRunResult = myBot.run(
            f"Read the /{DOCKER_WORKING_DIR}/calculator/calculator.log file and identify the error. "
            f"Then examine /{DOCKER_WORKING_DIR}/calculator/code/calculator.py and fix the divide function "
            f"to properly handle division by zero by adding a check before division and returning an appropriate message or value.",
            timeout_in_seconds=300,
        )
        
        # Print results (safely handle potential serialization issues)
        try:
            print(f"Status: {response.status}")
            print(f"***Result:***\n{response.result}")
            print(f"===\nError: {response.error}")
        except Exception as e:
            print(f"Could not print response due to serialization issue: {e}")
            try:
                status_str = str(response.status)
                result_str = str(response.result) if response.result is not None else "None"
                error_str = str(response.error) if response.error is not None else "None"
                print(f"Status: {status_str}")
                print(f"***Result:***\n{result_str}")
                print(f"===\nError: {error_str}")
            except Exception as e2:
                print(f"Status: {response.status}")
                print(f"Result type: {type(response.result)}")
                print(f"Error type: {type(response.error)}")
                print(f"Serialization error: {e2}")
        
        # Check if file was modified
        calc_file = calculator_data_dir / "code" / "calculator.py"
        if calc_file.exists():
            print(f"✅ calculator.py exists!")
            calc_content = calc_file.read_text()
            if "b == 0" in calc_content or "b != 0" in calc_content or "if not b" in calc_content:
                print(f"✅ Zero division check found in code!")
            else:
                print(f"⚠️ No obvious zero division check found")
            print(f"Updated content preview: {calc_content[:500]}...")
        else:
            print("❌ calculator.py was not found")
        
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
