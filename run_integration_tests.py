#!/usr/bin/env python3
"""
Test runner script for microbots integration tests.

This script demonstrates how to run the integration tests that simulate
user experience with the microbots library.
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """Run integration tests with different configurations."""

    # Get the project root directory
    project_root = Path(__file__).parent

    print("🤖 Microbots Integration Test Runner")
    print("=" * 50)

    # Check if we're in the right directory
    if not (project_root / "src" / "microbots").exists():
        print("❌ Error: Could not find microbots source code.")
        print("   Make sure you're running this from the project root directory.")
        sys.exit(1)

    # Check for .env file
    env_file = project_root / ".env"
    if not env_file.exists():
        print("⚠️  Warning: No .env file found.")
        print("   Tests requiring Azure OpenAI will be skipped.")
        print("   Create a .env file with:")
        print("     AZURE_OPENAI_API_KEY=your_key")
        print("     AZURE_OPENAI_ENDPOINT=your_endpoint")
        print("     AZURE_OPENAI_API_VERSION=your_version")
        print()

    # Different test run configurations
    test_configs = [
        {
            "name": "API Validation Tests (no API calls)",
            "args": [
                "tests/integration/test_microbots_api.py",
                "-m",
                "not requires_api",
            ],
            "description": "Tests that validate imports and API structure without making API calls",
        },
        {
            "name": "ReadingBot Integration Tests",
            "args": ["tests/integration/test_reading_bot_integration.py"],
            "description": "Full integration tests for ReadingBot (requires API keys)",
        },
        {
            "name": "WritingBot Integration Tests",
            "args": ["tests/integration/test_writing_bot_integration.py"],
            "description": "Full integration tests for WritingBot (requires API keys)",
        },
        {
            "name": "BrowsingBot Integration Tests",
            "args": ["tests/integration/test_browsing_bot_integration.py"],
            "description": "Full integration tests for BrowsingBot (requires API keys)",
        },
        {
            "name": "LogAnalysisBot Integration Tests",
            "args": ["tests/integration/test_log_analysis_bot_integration.py"],
            "description": "Full integration tests for LogAnalysisBot (requires API keys)",
        },
        {
            "name": "All Integration Tests",
            "args": ["tests/integration/"],
            "description": "Run all integration tests (API validation + full integration)",
        },
    ]

    print("Available test configurations:")
    for i, config in enumerate(test_configs, 1):
        print(f"  {i}. {config['name']}")
        print(f"     {config['description']}")

    print("\n" + "=" * 50)

    # Ask user which tests to run
    while True:
        try:
            choice = input(
                f"\nSelect test configuration (1-{len(test_configs)}) or 'q' to quit: "
            ).strip()

            if choice.lower() == "q":
                print("Goodbye! 👋")
                sys.exit(0)

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(test_configs):
                selected_config = test_configs[choice_idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(test_configs)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")

    print(f"\n🚀 Running: {selected_config['name']}")
    print(f"📝 Description: {selected_config['description']}")
    print("-" * 50)

    # Change to project root directory
    os.chdir(project_root)

    # Run the selected tests
    cmd = ["python", "-m", "pytest"] + selected_config["args"]
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(cmd, check=False)
        print(f"\n{'=' * 50}")
        if result.returncode == 0:
            print("✅ Tests completed successfully!")
        else:
            print("❌ Some tests failed or were skipped.")
            print("   Check the output above for details.")

        return result.returncode

    except KeyboardInterrupt:
        print("\n\n🛑 Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
