"""
Integration tests for ReadingBot that simulate user experience.

These tests demonstrate how users would import and use ReadingBot
from the microbots library.
"""

import pytest

# This is how users would import the library
from microbots import ReadingBot
from tests.integration.base_test import (
    MicrobotsIntegrationTestBase,
    requires_azure_openai,
)


class TestReadingBotIntegration(MicrobotsIntegrationTestBase):
    """Test ReadingBot from user perspective."""

    @requires_azure_openai
    def test_reading_bot_basic_usage(self):
        """Test basic ReadingBot usage as a user would."""
        # Create a test workspace with sample files
        workspace = self.create_test_workspace("reading_test")

        sample_files = {
            "README.md": """# My Project
            
This is a sample project for testing ReadingBot.

## Features
- Feature A: Does something useful
- Feature B: Does something else
- Feature C: Advanced functionality

## Installation
pip install my-project

## Usage
Run the main script to get started.
""",
            "main.py": """#!/usr/bin/env python3

def main():
    print("Hello, World!")
    return calculate_result()

def calculate_result():
    x = 10
    y = 20
    return x + y

if __name__ == "__main__":
    result = main()
    print(f"Result: {result}")
""",
            "config.json": """{
    "app_name": "My Test App",
    "version": "1.0.0",
    "debug": false,
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "testdb"
    }
}""",
        }

        self.create_sample_files(workspace, sample_files)

        # This is how a user would create and use ReadingBot
        reading_bot = ReadingBot(
            model="azure-openai/mini-swe-agent-gpt5", folder_to_mount=str(workspace)
        )

        # Test reading and understanding the codebase
        result = reading_bot.run(
            "What does this project do? Summarize the main functionality.",
            timeout_in_seconds=120,
        )

        # Verify the bot was able to read and understand the files
        assert result.status is True, f"ReadingBot failed: {result.error}"
        assert result.result is not None
        assert len(result.result.strip()) > 0

        print(f"ReadingBot result: {result.result}")

    @requires_azure_openai
    def test_reading_bot_specific_questions(self):
        """Test ReadingBot with specific technical questions."""
        workspace = self.create_test_workspace("reading_specific")

        sample_files = {
            "calculator.py": """class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def get_history(self):
        return self.history.copy()
""",
            "test_calculator.py": """import unittest
from calculator import Calculator

class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = Calculator()
    
    def test_add(self):
        self.assertEqual(self.calc.add(2, 3), 5)
    
    def test_multiply(self):
        self.assertEqual(self.calc.multiply(4, 5), 20)

if __name__ == "__main__":
    unittest.main()
""",
        }

        self.create_sample_files(workspace, sample_files)

        reading_bot = ReadingBot(
            model="azure-openai/mini-swe-agent-gpt5", folder_to_mount=str(workspace)
        )

        # Ask specific technical questions
        result = reading_bot.run(
            "What methods does the Calculator class have and what do they do?",
            timeout_in_seconds=120,
        )

        assert result.status is True, f"ReadingBot failed: {result.error}"
        assert "add" in result.result.lower()
        assert "multiply" in result.result.lower()

        print(f"Technical analysis result: {result.result}")

    def test_reading_bot_import_validation(self):
        """Test that ReadingBot can be imported correctly."""
        # This validates the public API
        from microbots import ReadingBot

        # Verify the class exists and has expected methods
        assert hasattr(ReadingBot, "__init__")
        assert hasattr(ReadingBot, "run")
        assert callable(getattr(ReadingBot, "run"))

    def test_reading_bot_initialization_parameters(self):
        """Test ReadingBot initialization with various parameters."""
        workspace = self.create_test_workspace("init_test")

        # Test basic initialization
        bot1 = ReadingBot(
            model="azure-openai/test-model", folder_to_mount=str(workspace)
        )
        assert bot1 is not None

        # Test initialization with additional parameters if they exist
        try:
            bot2 = ReadingBot(
                model="azure-openai/test-model",
                folder_to_mount=str(workspace),
                # Add any optional parameters that might exist
            )
            assert bot2 is not None
        except TypeError:
            # If additional parameters don't exist, that's fine
            pass
