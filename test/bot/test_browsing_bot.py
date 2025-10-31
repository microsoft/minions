import logging
import os
import sys
import pytest

# Setup logging for tests
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/")))
from microbots import BrowsingBot, BotRunResult

@pytest.mark.integration
@pytest.mark.docker
@pytest.mark.slow
class TestBrowsingBot:
    """Integration tests for BrowsingBot functionality."""
    
    @pytest.fixture(scope="class")
    def browsing_bot(self):
        """Create a BrowsingBot instance for testing."""
        bot = BrowsingBot(model="azure-openai/mini-swe-agent-gpt5")
        yield bot
        # Cleanup: stop the environment
        if hasattr(bot, 'environment') and bot.environment:
            try:
                bot.environment.stop()
            except Exception as e:
                logger.warning(f"Error stopping environment: {e}")
    
    def test_simple_question_response(self, browsing_bot):
        """Test that the bot can answer a simple factual question."""
        response: BotRunResult = browsing_bot.run(
            "What is the capital of France?",
            timeout_in_seconds=300,
        )
        
        # Assert the response was successful
        assert response.status == True, f"Bot failed with error: {response.error}"
        assert response.result is not None, "Bot returned no result"
        assert isinstance(response.result, str), "Result should be a string"
        
        # Check that the result contains the expected answer
        result_lower = response.result.lower()
        assert "paris" in result_lower, f"Expected 'Paris' in result, got: {response.result}"
        
        logger.info(f"Test passed. Bot response: {response.result}")
    
    
    @pytest.mark.parametrize("query,expected_keywords", [
        ("What is the capital of Germany?", ["berlin"]),
        ("What is 2+2?", ["4", "four"]),
        ("Who is the current President of the United States?", ["Trump"]),
    ])
    def test_multiple_queries(self, browsing_bot, query, expected_keywords):
        """Test the bot with multiple different queries."""
        response: BotRunResult = browsing_bot.run(query, timeout_in_seconds=300)
        
        assert response.status == True, f"Query '{query}' failed: {response.error}"
        assert response.result is not None, f"No result for query: {query}"
        
        result_lower = response.result.lower()
        # At least one expected keyword should be in the result
        keyword_found = any(keyword.lower() in result_lower for keyword in expected_keywords)
        assert keyword_found, f"None of {expected_keywords} found in result: {response.result}"
        
        logger.info(f"Query '{query}' passed with result: {response.result[:100]}...")
