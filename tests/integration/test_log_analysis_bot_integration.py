"""
Integration tests for LogAnalysisBot that simulate user experience.

These tests demonstrate how users would import and use LogAnalysisBot
from the microbots library.
"""

# This is how users would import the library
from microbots import LogAnalysisBot
from tests.integration.base_test import (
    MicrobotsIntegrationTestBase,
    requires_azure_openai,
)


class TestLogAnalysisBotIntegration(MicrobotsIntegrationTestBase):
    """Test LogAnalysisBot from user perspective."""

    @requires_azure_openai
    def test_log_analysis_bot_basic_usage(self):
        """Test basic LogAnalysisBot usage as a user would."""
        # Create a test workspace with sample log files
        workspace = self.create_test_workspace("log_analysis_test")

        sample_files = {
            "app.log": """2024-09-29 10:15:23 INFO  [main] Application started successfully
2024-09-29 10:15:24 INFO  [DatabaseConnection] Connected to database: localhost:5432
2024-09-29 10:16:45 WARN  [UserService] User authentication took longer than expected: 2.3s
2024-09-29 10:17:12 ERROR [PaymentProcessor] Payment processing failed for order #12345: Invalid credit card
2024-09-29 10:17:13 ERROR [PaymentProcessor] Retrying payment for order #12345
2024-09-29 10:17:15 INFO  [PaymentProcessor] Payment successful for order #12345
2024-09-29 10:18:30 WARN  [MemoryMonitor] Memory usage high: 85%
2024-09-29 10:19:45 ERROR [DatabaseConnection] Connection timeout to localhost:5432
2024-09-29 10:19:46 INFO  [DatabaseConnection] Reconnected to database successfully
2024-09-29 10:20:00 INFO  [main] Application running normally
""",
            "error.log": """2024-09-29 10:17:12 ERROR [PaymentProcessor] Payment processing failed for order #12345: Invalid credit card
java.lang.IllegalArgumentException: Credit card number is invalid
    at com.example.payment.CreditCardValidator.validate(CreditCardValidator.java:45)
    at com.example.payment.PaymentProcessor.processPayment(PaymentProcessor.java:123)
    at com.example.service.OrderService.completeOrder(OrderService.java:89)

2024-09-29 10:19:45 ERROR [DatabaseConnection] Connection timeout to localhost:5432
org.postgresql.util.PSQLException: Connection timeout
    at org.postgresql.core.PGStream.receiveChar(PGStream.java:190)
    at org.postgresql.core.ConnectionFactory.openConnection(ConnectionFactory.java:123)
""",
        }

        self.create_sample_files(workspace, sample_files)

        # This is how a user would create and use LogAnalysisBot
        log_analysis_bot = LogAnalysisBot(
            model="azure-openai/mini-swe-agent-gpt5", folder_to_mount=str(workspace)
        )

        # Test log analysis functionality
        result = log_analysis_bot.run(
            "Analyze the logs and identify the main issues and patterns. What problems occurred?",
            timeout_in_seconds=120,
        )

        # Verify the bot was able to analyze the logs
        assert result.status is True, f"LogAnalysisBot failed: {result.error}"
        assert result.result is not None
        assert len(str(result.result).strip()) > 0

        result_text = str(result.result).lower()
        # Should identify key issues from the logs
        assert any(
            term in result_text
            for term in ["payment", "database", "connection", "error", "timeout"]
        )

        print(f"LogAnalysisBot result: {result.result}")

    @requires_azure_openai
    def test_log_analysis_bot_error_patterns(self):
        """Test LogAnalysisBot identifying error patterns."""
        workspace = self.create_test_workspace("log_patterns")

        sample_files = {
            "service.log": """2024-09-29 08:00:01 INFO  [ServiceA] Service started
2024-09-29 08:15:23 ERROR [ServiceA] OutOfMemoryError: Java heap space exceeded
2024-09-29 08:15:24 INFO  [ServiceA] Service restarted
2024-09-29 08:30:45 ERROR [ServiceA] OutOfMemoryError: Java heap space exceeded  
2024-09-29 08:30:46 INFO  [ServiceA] Service restarted
2024-09-29 08:45:12 ERROR [ServiceA] OutOfMemoryError: Java heap space exceeded
2024-09-29 08:45:13 INFO  [ServiceA] Service restarted
2024-09-29 09:00:30 ERROR [ServiceA] OutOfMemoryError: Java heap space exceeded
2024-09-29 09:00:31 INFO  [ServiceA] Service restarted
2024-09-29 09:15:45 WARN  [ServiceA] Memory usage: 95%
2024-09-29 09:16:00 ERROR [ServiceA] OutOfMemoryError: Java heap space exceeded
""",
            "analysis_request.txt": "Focus on recurring errors and memory issues",
        }

        self.create_sample_files(workspace, sample_files)

        log_analysis_bot = LogAnalysisBot(
            model="azure-openai/mini-swe-agent-gpt5", folder_to_mount=str(workspace)
        )

        # Ask for specific pattern analysis
        result = log_analysis_bot.run(
            "Identify recurring error patterns and suggest potential solutions for the memory issues.",
            timeout_in_seconds=120,
        )

        assert result.status is True, f"LogAnalysisBot failed: {result.error}" ""

        result_text = str(result.result).lower()
        # Should identify the memory pattern
        assert any(
            term in result_text
            for term in ["memory", "heap", "recurring", "pattern", "outofmemory"]
        )

        print(f"Pattern analysis result: {result.result}")

    def test_log_analysis_bot_import_validation(self):
        """Test that LogAnalysisBot can be imported correctly."""
        # This validates the public API
        from microbots import LogAnalysisBot

        # Verify the class exists and has expected methods
        assert hasattr(LogAnalysisBot, "__init__")
        assert hasattr(LogAnalysisBot, "run")
        assert callable(getattr(LogAnalysisBot, "run"))

    @requires_azure_openai
    def test_log_analysis_bot_performance_analysis(self):
        """Test LogAnalysisBot analyzing performance logs."""
        workspace = self.create_test_workspace("performance_logs")

        sample_files = {
            "performance.log": """2024-09-29 10:00:00 INFO  [API] GET /api/users - 45ms - 200
2024-09-29 10:00:05 INFO  [API] POST /api/orders - 123ms - 201  
2024-09-29 10:00:10 INFO  [API] GET /api/products - 2340ms - 200
2024-09-29 10:00:15 WARN  [API] GET /api/products - 3450ms - 200 (SLOW)
2024-09-29 10:00:20 INFO  [API] DELETE /api/orders/123 - 67ms - 204
2024-09-29 10:00:25 INFO  [API] GET /api/products - 2890ms - 200
2024-09-29 10:00:30 ERROR [API] GET /api/products - 5000ms - 500 (TIMEOUT)
2024-09-29 10:00:35 INFO  [API] GET /api/users - 52ms - 200
2024-09-29 10:00:40 WARN  [API] GET /api/products - 4100ms - 200 (SLOW)
""",
            "metrics.log": """Memory Usage: 78%
CPU Usage: 45%
Database Connections: 15/20
Active Users: 234
Cache Hit Rate: 67%
""",
        }

        self.create_sample_files(workspace, sample_files)

        log_analysis_bot = LogAnalysisBot(
            model="azure-openai/mini-swe-agent-gpt5", folder_to_mount=str(workspace)
        )

        # Analyze performance issues
        result = log_analysis_bot.run(
            "Analyze the performance logs and identify which API endpoints are having issues. What recommendations do you have?",
            timeout_in_seconds=120,
        )

        assert result.status is True, f"LogAnalysisBot failed: {result.error}"

        result_text = str(result.result).lower()
        # Should identify the /api/products endpoint issues
        assert any(
            term in result_text
            for term in ["products", "slow", "timeout", "performance", "api"]
        )

        print(f"Performance analysis result: {result.result}")

    def test_log_analysis_bot_initialization_parameters(self):
        """Test LogAnalysisBot initialization with various parameters."""
        workspace = self.create_test_workspace("init_test")

        # Test basic initialization
        bot1 = LogAnalysisBot(
            model="azure-openai/test-model", folder_to_mount=str(workspace)
        )
        assert bot1 is not None

        # Test initialization with additional parameters if they exist
        try:
            bot2 = LogAnalysisBot(
                model="azure-openai/test-model",
                folder_to_mount=str(workspace),
                # Add any optional parameters that might exist
            )
            assert bot2 is not None
        except TypeError:
            # If additional parameters don't exist, that's fine
            pass
