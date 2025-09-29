"""
Integration tests for BrowsingBot that simulate user experience.

These tests demonstrate how users would import and use BrowsingBot
from the microbots library.
"""

# This is how users would import the library
from microbots import BrowsingBot
from tests.integration.base_test import (
    MicrobotsIntegrationTestBase,
    requires_azure_openai,
)


class TestBrowsingBotIntegration(MicrobotsIntegrationTestBase):
    """Test BrowsingBot from user perspective."""

    @requires_azure_openai
    def test_browsing_bot_basic_usage(self):
        """Test basic BrowsingBot usage as a user would."""
        # This is how a user would create and use BrowsingBot
        browsing_bot = BrowsingBot(model="azure-openai/mini-swe-agent-gpt5")

        # Test basic web search functionality
        result = browsing_bot.run(
            "What is the capital of France?", timeout_in_seconds=120
        )

        # Verify the bot was able to browse and get information
        assert result.status is True, f"BrowsingBot failed: {result.error}"
        assert result.result is not None
        assert len(str(result.result).strip()) > 0

        # The result should contain information about Paris
        result_text = str(result.result).lower()
        assert "paris" in result_text, f"Expected 'paris' in result: {result.result}"

        print(f"BrowsingBot result: {result.result}")

    @requires_azure_openai
    def test_browsing_bot_technical_search(self):
        """Test BrowsingBot with technical queries."""
        browsing_bot = BrowsingBot(model="azure-openai/mini-swe-agent-gpt5")

        # Test technical information search
        result = browsing_bot.run(
            "What is the latest stable version of Python?", timeout_in_seconds=150
        )

        assert result.status is True, f"BrowsingBot failed: {result.error}"
        assert result.result is not None

        result_text = str(result.result).lower()
        # Should contain some version number or python-related information
        assert (
            "python" in result_text or "3." in result_text or "version" in result_text
        )

        print(f"Technical search result: {result.result}")

    def test_browsing_bot_import_validation(self):
        """Test that BrowsingBot can be imported correctly."""
        # This validates the public API
        from microbots import BrowsingBot

        # Verify the class exists and has expected methods
        assert hasattr(BrowsingBot, "__init__")
        assert hasattr(BrowsingBot, "run")
        assert callable(getattr(BrowsingBot, "run"))

    @requires_azure_openai
    def test_browsing_bot_current_events(self):
        """Test BrowsingBot with current events queries."""
        browsing_bot = BrowsingBot(model="azure-openai/mini-swe-agent-gpt5")

        # Test current events search (this might be more variable in results)
        result = browsing_bot.run(
            "What are the current trending topics in AI and machine learning?",
            timeout_in_seconds=180,
        )

        assert result.status is True, f"BrowsingBot failed: {result.error}"
        assert result.result is not None

        result_text = str(result.result).lower()
        # Should contain AI or ML related terms
        assert any(
            term in result_text
            for term in [
                "ai",
                "artificial intelligence",
                "machine learning",
                "ml",
                "technology",
            ]
        )

        print(f"Current events result: {result.result}")

    def test_browsing_bot_initialization_parameters(self):
        """Test BrowsingBot initialization with various parameters."""
        # Test basic initialization
        bot1 = BrowsingBot(model="azure-openai/test-model")
        assert bot1 is not None

        # Test initialization with additional tools if supported
        try:
            bot2 = BrowsingBot(model="azure-openai/test-model", additional_tools=[])
            assert bot2 is not None
        except TypeError:
            # If additional_tools parameter doesn't exist in the signature, that's fine
            pass
