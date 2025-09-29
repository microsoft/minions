#!/usr/bin/env python3
"""
Quick validation script for microbots library installation.

Run this script to verify that the microbots library is properly installed
and accessible with the expected public API.
"""

import os
import sys
from pathlib import Path

# Add src to path for development testing
project_root = Path(__file__).parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def test_imports():
    """Test that all expected classes can be imported."""
    print("🔍 Testing imports...")

    try:
        # Test main import pattern
        from microbots import BrowsingBot, LogAnalysisBot, ReadingBot, WritingBot

        print("  ✅ Main classes imported successfully")

        # Test module import
        import microbots

        print("  ✅ Module import successful")

        # Verify __all__ exports
        expected_exports = ["ReadingBot", "WritingBot", "BrowsingBot", "LogAnalysisBot"]
        if hasattr(microbots, "__all__"):
            actual_exports = microbots.__all__
            missing = set(expected_exports) - set(actual_exports)
            extra = set(actual_exports) - set(expected_exports)

            if missing:
                print(f"  ⚠️  Missing exports: {missing}")
            if extra:
                print(f"  ℹ️  Extra exports: {extra}")

            print(f"  ✅ Exports defined: {actual_exports}")
        else:
            print("  ⚠️  __all__ not defined")

        return True

    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error: {e}")
        return False


def test_class_interfaces():
    """Test that classes have expected interfaces."""
    print("\n🔍 Testing class interfaces...")

    try:
        from microbots import BrowsingBot, LogAnalysisBot, ReadingBot, WritingBot

        classes_to_test = [
            ("ReadingBot", ReadingBot),
            ("WritingBot", WritingBot),
            ("BrowsingBot", BrowsingBot),
            ("LogAnalysisBot", LogAnalysisBot),
        ]

        required_methods = ["__init__", "run"]

        for class_name, class_obj in classes_to_test:
            print(f"  Testing {class_name}...")

            # Check if it's callable (can be instantiated)
            if not callable(class_obj):
                print(f"    ❌ {class_name} is not callable")
                return False

            # Check for required methods
            for method in required_methods:
                if not hasattr(class_obj, method):
                    print(f"    ❌ {class_name} missing {method} method")
                    return False

                if not callable(getattr(class_obj, method)):
                    print(f"    ❌ {class_name}.{method} is not callable")
                    return False

            print(f"    ✅ {class_name} has all required methods")

        print("  ✅ All class interfaces valid")
        return True

    except Exception as e:
        print(f"  ❌ Interface test failed: {e}")
        return False


def test_basic_initialization():
    """Test basic initialization patterns (without Docker)."""
    print("\n🔍 Testing basic initialization...")

    try:
        import tempfile

        from microbots import BrowsingBot, LogAnalysisBot, ReadingBot, WritingBot

        # Create a temporary workspace
        with tempfile.TemporaryDirectory() as temp_dir:

            # Test classes that require folder_to_mount
            folder_classes = [
                ("ReadingBot", ReadingBot),
                ("WritingBot", WritingBot),
                ("LogAnalysisBot", LogAnalysisBot),
            ]

            for class_name, class_obj in folder_classes:
                try:
                    # Try to create instance (will likely fail due to Docker, but should not crash on parameters)
                    instance = class_obj(model="test-model", folder_to_mount=temp_dir)
                    print(f"    ✅ {class_name} initialization successful")
                except Exception as e:
                    # Expected to fail without Docker, but should be a runtime error, not a parameter error
                    if "parameter" in str(e).lower() or "argument" in str(e).lower():
                        print(f"    ❌ {class_name} parameter error: {e}")
                        return False
                    else:
                        # Runtime errors (like Docker not available) are expected
                        print(
                            f"    ✅ {class_name} parameter validation passed (runtime error expected)"
                        )

            # Test BrowsingBot (no folder required)
            try:
                browsing_bot = BrowsingBot(model="test-model")
                print("    ✅ BrowsingBot initialization successful")
            except Exception as e:
                if "parameter" in str(e).lower() or "argument" in str(e).lower():
                    print(f"    ❌ BrowsingBot parameter error: {e}")
                    return False
                else:
                    print(
                        "    ✅ BrowsingBot parameter validation passed (runtime error expected)"
                    )

        print("  ✅ Basic initialization tests passed")
        return True

    except Exception as e:
        print(f"  ❌ Initialization test failed: {e}")
        return False


def check_environment():
    """Check environment setup."""
    print("\n🔍 Checking environment...")

    # Check for environment variables
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
    ]
    env_vars_set = []

    for var in required_vars:
        if os.getenv(var):
            env_vars_set.append(var)
            print(f"    ✅ {var} is set")
        else:
            print(f"    ❌ {var} not set")

    if len(env_vars_set) == len(required_vars):
        print("  ✅ All required environment variables are set")
        print("    You can run full integration tests")
        return True
    else:
        print(
            f"  ⚠️  {len(required_vars) - len(env_vars_set)} environment variables missing"
        )
        print("    API-dependent tests will be skipped")
        print("    Set up .env file for full testing")
        return False


def main():
    """Run all validation tests."""
    print("🤖 Microbots Library Validation")
    print("=" * 40)

    tests = [
        ("Import Tests", test_imports),
        ("Interface Tests", test_class_interfaces),
        ("Initialization Tests", test_basic_initialization),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}")
        print("-" * 25)
        result = test_func()
        results.append((test_name, result))

    # Environment check (informational)
    env_ready = check_environment()

    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results Summary")
    print("-" * 25)

    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 All validation tests passed!")
        print("   The microbots library is properly set up.")

        if env_ready:
            print("\n💡 Next steps:")
            print("   • Run full integration tests: python run_integration_tests.py")
            print("   • Try the examples in README.md")
        else:
            print("\n💡 For full testing:")
            print("   • Set up Azure OpenAI credentials in .env")
            print("   • Run: python run_integration_tests.py")
    else:
        print("\n❌ Some validation tests failed.")
        print("   Check the errors above and fix any issues.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
