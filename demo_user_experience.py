#!/usr/bin/env python3
"""
Example usage script demonstrating how users would interact with the microbots library.

This script shows the typical user journey from installation to usage.
"""

import os
import sys
from pathlib import Path

# Add src to path to simulate installed package
project_root = Path(__file__).parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Load environment variables
from dotenv import load_dotenv

load_dotenv()


def demonstrate_reading_bot():
    """Demonstrate ReadingBot usage."""
    print("\n📖 ReadingBot Example")
    print("-" * 30)

    # This is how users would import and use ReadingBot
    from microbots import ReadingBot

    print("✅ Successfully imported ReadingBot")
    print("📁 Creating test workspace...")

    # Create a sample project
    import tempfile

    workspace = Path(tempfile.mkdtemp(prefix="microbots_demo_"))

    # Create sample files
    (workspace / "README.md").write_text(
        """# Demo Project
    
This is a sample Python project for demonstrating ReadingBot.

## Features
- Data processing
- File management  
- API integration
"""
    )

    (workspace / "main.py").write_text(
        """#!/usr/bin/env python3

def process_data(data):
    \"\"\"Process input data and return results.\"\"\"
    return [item.upper() for item in data if item]

def main():
    sample_data = ["hello", "world", "", "python"]
    result = process_data(sample_data)
    print("Processed data:", result)

if __name__ == "__main__":
    main()
"""
    )

    print(f"📂 Created workspace at: {workspace}")
    print("📄 Sample files created: README.md, main.py")

    # Initialize ReadingBot (but don't run it without API keys)
    try:
        reading_bot = ReadingBot(
            model="azure-openai/gpt-4", folder_to_mount=str(workspace)
        )
        print("✅ ReadingBot initialized successfully")
        print("   (Note: Actual execution requires API keys)")

        # Show what a user would do next:
        print("\n💡 Next steps for users:")
        print("   result = reading_bot.run('What does this project do?')")
        print("   print(result.result)")

    except Exception as e:
        print(f"ℹ️  ReadingBot initialization: {e}")

    # Cleanup
    import shutil

    shutil.rmtree(workspace, ignore_errors=True)


def demonstrate_writing_bot():
    """Demonstrate WritingBot usage."""
    print("\n✏️  WritingBot Example")
    print("-" * 30)

    from microbots import WritingBot

    print("✅ Successfully imported WritingBot")

    # Create sample workspace
    import tempfile

    workspace = Path(tempfile.mkdtemp(prefix="microbots_writing_"))

    (workspace / "package.json").write_text(
        """{
  "name": "demo-app",
  "version": "1.0.0",
  "description": "Demo application",
  "main": "index.js",
  "scripts": {
    "start": "node index.js"
  }
}"""
    )

    print(f"📂 Created workspace at: {workspace}")

    try:
        writing_bot = WritingBot(
            model="azure-openai/gpt-4", folder_to_mount=str(workspace)
        )
        print("✅ WritingBot initialized successfully")

        print("\n💡 Example usage:")
        print("   result = writing_bot.run('Add a simple express server to index.js')")
        print("   print(result.result)")

    except Exception as e:
        print(f"ℹ️  WritingBot initialization: {e}")

    # Cleanup
    import shutil

    shutil.rmtree(workspace, ignore_errors=True)


def demonstrate_browsing_bot():
    """Demonstrate BrowsingBot usage."""
    print("\n🌐 BrowsingBot Example")
    print("-" * 30)

    from microbots import BrowsingBot

    print("✅ Successfully imported BrowsingBot")

    try:
        browsing_bot = BrowsingBot(model="azure-openai/gpt-4")
        print("✅ BrowsingBot initialized successfully")

        print("\n💡 Example usage:")
        print("   result = browsing_bot.run('What is the latest Python version?')")
        print("   print(result.result)")

    except Exception as e:
        print(f"ℹ️  BrowsingBot initialization: {e}")


def demonstrate_log_analysis_bot():
    """Demonstrate LogAnalysisBot usage."""
    print("\n📊 LogAnalysisBot Example")
    print("-" * 35)

    from microbots import LogAnalysisBot

    print("✅ Successfully imported LogAnalysisBot")

    # Create sample workspace with logs
    import tempfile

    workspace = Path(tempfile.mkdtemp(prefix="microbots_logs_"))

    (workspace / "app.log").write_text(
        """2024-09-29 10:00:00 INFO  Application started
2024-09-29 10:01:23 ERROR Database connection failed
2024-09-29 10:01:24 WARN  Retrying database connection
2024-09-29 10:01:25 INFO  Database connected successfully
2024-09-29 10:05:45 ERROR OutOfMemoryError in UserService
2024-09-29 10:05:46 INFO  Service restarted
"""
    )

    print(f"📂 Created workspace with sample logs at: {workspace}")

    try:
        log_bot = LogAnalysisBot(
            model="azure-openai/gpt-4", folder_to_mount=str(workspace)
        )
        print("✅ LogAnalysisBot initialized successfully")

        print("\n💡 Example usage:")
        print("   result = log_bot.run('Analyze the logs and identify issues')")
        print("   print(result.result)")

    except Exception as e:
        print(f"ℹ️  LogAnalysisBot initialization: {e}")

    # Cleanup
    import shutil

    shutil.rmtree(workspace, ignore_errors=True)


def check_environment():
    """Check if the environment is properly set up."""
    print("🔍 Environment Check")
    print("-" * 20)

    # Check for required environment variables
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
    ]

    env_status = {}
    for var in required_vars:
        value = os.getenv(var)
        env_status[var] = "✅ Set" if value else "❌ Not set"
        print(f"  {var}: {env_status[var]}")

    all_set = all("✅" in status for status in env_status.values())

    if all_set:
        print("\n✅ All required environment variables are set!")
        print("   You can run the full integration tests.")
    else:
        print("\n⚠️  Some environment variables are missing.")
        print("   API-dependent tests will be skipped.")
        print("   To run full tests, create a .env file with the missing variables.")

    return all_set


def main():
    """Main demonstration function."""
    print("🤖 Microbots Library User Experience Demo")
    print("=" * 50)

    # Check environment
    env_ready = check_environment()

    print("\n📦 Testing Library Imports")
    print("-" * 30)

    try:
        # Test main import patterns users would use
        from microbots import BrowsingBot, LogAnalysisBot, ReadingBot, WritingBot

        print("✅ All main classes imported successfully!")

        # Test alternative import
        import microbots

        print("✅ Module-level import successful!")
        print(f"   Available classes: {microbots.__all__}")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return 1

    # Demonstrate each bot
    demonstrate_reading_bot()
    demonstrate_writing_bot()
    demonstrate_browsing_bot()
    demonstrate_log_analysis_bot()

    print("\n" + "=" * 50)
    print("🎉 Demo completed successfully!")

    if env_ready:
        print("\n💡 Next steps:")
        print("   1. Run: python run_integration_tests.py")
        print("   2. Choose test configuration to validate functionality")
    else:
        print("\n💡 To run full tests:")
        print("   1. Set up Azure OpenAI credentials in .env file")
        print("   2. Run: python run_integration_tests.py")

    print("\n📚 For more examples, check:")
    print("   - tests/integration/ directory")
    print("   - README.md file")

    return 0


if __name__ == "__main__":
    sys.exit(main())
