"""
Integration tests for WritingBot that simulate user experience.

These tests demonstrate how users would import and use WritingBot
from the microbots library.
"""

# This is how users would import the library
from microbots import WritingBot
from tests.integration.base_test import (
    MicrobotsIntegrationTestBase,
    requires_azure_openai,
)


class TestWritingBotIntegration(MicrobotsIntegrationTestBase):
    """Test WritingBot from user perspective."""

    @requires_azure_openai
    def test_writing_bot_basic_usage(self):
        """Test basic WritingBot usage as a user would."""
        # Create a test workspace
        workspace = self.create_test_workspace("writing_test")

        # Create a basic project structure
        sample_files = {
            "package.json": """{
  "name": "my-app",
  "version": "1.0.0",
  "description": "Test application",
  "main": "index.js",
  "scripts": {
    "start": "node index.js",
    "test": "echo \\"Error: no test specified\\" && exit 1"
  }
}""",
            "index.js": """console.log("Hello World!");
// TODO: Add more functionality here
""",
        }

        self.create_sample_files(workspace, sample_files)

        # This is how a user would create and use WritingBot
        writing_bot = WritingBot(
            model="azure-openai/mini-swe-agent-gpt5", folder_to_mount=str(workspace)
        )

        # Test writing new functionality
        result = writing_bot.run(
            "Add a simple HTTP server to index.js that serves 'Hello from server!' on port 3000",
            timeout_in_seconds=180,
        )

        # Verify the bot was able to make changes
        assert result.status is True, f"WritingBot failed: {result.error}"
        assert result.result is not None

        # Check if the file was modified
        updated_index = (workspace / "index.js").read_text()
        assert "server" in updated_index.lower() or "http" in updated_index.lower()

        print(f"WritingBot result: {result.result}")

    @requires_azure_openai
    def test_writing_bot_bug_fix(self):
        """Test WritingBot fixing bugs in existing code."""
        workspace = self.create_test_workspace("writing_bugfix")

        sample_files = {
            "calculator.py": """def divide(a, b):
    # This has a bug - no division by zero check
    return a / b

def calculate_average(numbers):
    # This has a bug - doesn't handle empty list
    return sum(numbers) / len(numbers)

def main():
    print(divide(10, 0))  # This will crash
    print(calculate_average([]))  # This will also crash
    
if __name__ == "__main__":
    main()
""",
            "README.md": """# Calculator App

This app has some bugs that need to be fixed:
1. Division by zero error
2. Empty list handling in average calculation
""",
        }

        self.create_sample_files(workspace, sample_files)

        writing_bot = WritingBot(
            model="azure-openai/mini-swe-agent-gpt5", folder_to_mount=str(workspace)
        )

        # Ask the bot to fix the bugs
        result = writing_bot.run(
            "Fix the bugs in calculator.py. Add proper error handling for division by zero and empty list in average calculation.",
            timeout_in_seconds=180,
        )

        assert result.status is True, f"WritingBot failed: {result.error}"

        # Check if the fixes were applied
        fixed_code = (workspace / "calculator.py").read_text()
        assert (
            "zero" in fixed_code.lower() or "0" in fixed_code
        )  # Some form of zero check
        assert (
            "empty" in fixed_code.lower() or "len(" in fixed_code
        )  # Some form of empty check

        print(f"Bug fix result: {result.result}")

    def test_writing_bot_import_validation(self):
        """Test that WritingBot can be imported correctly."""
        # This validates the public API
        from microbots import WritingBot

        # Verify the class exists and has expected methods
        assert hasattr(WritingBot, "__init__")
        assert hasattr(WritingBot, "run")
        assert callable(getattr(WritingBot, "run"))

    @requires_azure_openai
    def test_writing_bot_create_new_files(self):
        """Test WritingBot creating entirely new files."""
        workspace = self.create_test_workspace("writing_new_files")

        # Start with minimal project
        sample_files = {"requirements.txt": "requests==2.28.0\n"}

        self.create_sample_files(workspace, sample_files)

        writing_bot = WritingBot(
            model="azure-openai/mini-swe-agent-gpt5", folder_to_mount=str(workspace)
        )

        # Ask bot to create new files
        result = writing_bot.run(
            "Create a simple Python web scraper script called 'scraper.py' that fetches and prints the title of a webpage using requests library.",
            timeout_in_seconds=180,
        )

        assert result.status is True, f"WritingBot failed: {result.error}"

        # Check if new file was created
        scraper_file = workspace / "scraper.py"
        assert scraper_file.exists(), "scraper.py should have been created"

        scraper_content = scraper_file.read_text()
        assert "requests" in scraper_content
        assert "title" in scraper_content.lower()

        print(f"New file creation result: {result.result}")
