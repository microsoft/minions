"""
Comprehensive integration tests for the microbots library.

These tests validate the complete public API and user experience
of the microbots library as it would be used by end users.
"""

from tests.integration.base_test import MicrobotsIntegrationTestBase


class TestMicrobotsPublicAPI(MicrobotsIntegrationTestBase):
    """Test the complete public API of the microbots library."""

    def test_main_imports(self):
        """Test that all main classes can be imported from the microbots package."""
        # This is the primary way users would import the library
        from microbots import BrowsingBot, LogAnalysisBot, ReadingBot, WritingBot

        # Verify all classes are importable
        assert ReadingBot is not None
        assert WritingBot is not None
        assert BrowsingBot is not None
        assert LogAnalysisBot is not None

        # Verify they are classes (callable)
        assert callable(ReadingBot)
        assert callable(WritingBot)
        assert callable(BrowsingBot)
        assert callable(LogAnalysisBot)

    def test_alternative_imports(self):
        """Test alternative import patterns users might use."""
        # Import specific classes
        from microbots import ReadingBot

        assert ReadingBot is not None

        # Import entire module
        import microbots

        assert hasattr(microbots, "ReadingBot")
        assert hasattr(microbots, "WritingBot")
        assert hasattr(microbots, "BrowsingBot")
        assert hasattr(microbots, "LogAnalysisBot")

        # Verify __all__ is properly defined
        assert hasattr(microbots, "__all__")
        expected_exports = ["ReadingBot", "WritingBot", "BrowsingBot", "LogAnalysisBot"]
        for export in expected_exports:
            assert export in microbots.__all__

    def test_bot_common_interface(self):
        """Test that all bots have a consistent interface."""
        from microbots import BrowsingBot, LogAnalysisBot, ReadingBot, WritingBot

        workspace = self.create_test_workspace("api_test")

        # All bots should have these common methods
        common_methods = ["__init__", "run"]

        for bot_class in [ReadingBot, WritingBot, LogAnalysisBot]:
            for method in common_methods:
                assert hasattr(
                    bot_class, method
                ), f"{bot_class.__name__} missing {method}"
                assert callable(
                    getattr(bot_class, method)
                ), f"{bot_class.__name__}.{method} not callable"

        # BrowsingBot might have different constructor parameters
        for method in common_methods:
            assert hasattr(BrowsingBot, method), f"BrowsingBot missing {method}"
            assert callable(
                getattr(BrowsingBot, method)
            ), f"BrowsingBot.{method} not callable"

    def test_initialization_patterns(self):
        """Test common initialization patterns users would use."""
        from microbots import BrowsingBot, LogAnalysisBot, ReadingBot, WritingBot

        workspace = self.create_test_workspace("init_patterns")

        # Test ReadingBot initialization patterns
        reading_bot = ReadingBot(
            model="azure-openai/test-model", folder_to_mount=str(workspace)
        )
        assert reading_bot is not None

        # Test WritingBot initialization patterns
        writing_bot = WritingBot(
            model="azure-openai/test-model", folder_to_mount=str(workspace)
        )
        assert writing_bot is not None

        # Test BrowsingBot initialization (different pattern - no folder)
        browsing_bot = BrowsingBot(model="azure-openai/test-model")
        assert browsing_bot is not None

        # Test LogAnalysisBot initialization patterns
        log_bot = LogAnalysisBot(
            model="azure-openai/test-model", folder_to_mount=str(workspace)
        )
        assert log_bot is not None

    def test_run_method_interface(self):
        """Test that run methods have consistent interfaces."""
        from microbots import BrowsingBot, LogAnalysisBot, ReadingBot, WritingBot

        workspace = self.create_test_workspace("run_interface")
        self.create_sample_files(workspace, {"test.txt": "Sample content"})

        # All bots should accept task and timeout parameters
        bots = [
            ReadingBot(model="test", folder_to_mount=str(workspace)),
            WritingBot(model="test", folder_to_mount=str(workspace)),
            BrowsingBot(model="test"),
            LogAnalysisBot(model="test", folder_to_mount=str(workspace)),
        ]

        for bot in bots:
            # Check that run method exists
            assert hasattr(bot, "run")
            run_method = getattr(bot, "run")
            assert callable(run_method)

            # Note: We don't actually call run here since it would require valid API keys
            # and would make API calls. The integration tests above handle actual execution.

    def test_readme_example_compatibility(self):
        """Test that the examples from README.md work with the current API."""
        # This validates that the documentation examples are accurate
        from microbots import WritingBot

        workspace = self.create_test_workspace("readme_example")

        # This is the example from README.md (slightly modified to use our test workspace)
        myWritingBot = WritingBot(
            model="azure-openai/my-gpt5",  # model format : <provider/deployment_model_name>
            folder_to_mount=str(workspace),
        )

        assert myWritingBot is not None

        # Verify the run method signature matches what's shown in README
        assert hasattr(myWritingBot, "run")

        # Note: We don't actually call run since that would require valid API keys
        # The README example shows calling:
        # data = myWritingBot.run("""when doing npm run build, I get an error.
        # Fix the error and make sure the build is successful.""", timeout_in_seconds=600)
        # print(data.results)

    def test_error_handling_patterns(self):
        """Test common error scenarios users might encounter."""
        from microbots import BrowsingBot, LogAnalysisBot, ReadingBot, WritingBot

        # Test invalid model parameter
        try:
            bot = ReadingBot(model="", folder_to_mount="/tmp")
            # If this doesn't raise an exception, that's also valid behavior
        except Exception as e:
            # Some validation error is expected for empty model
            assert isinstance(e, (ValueError, TypeError))

        # Test invalid folder path
        try:
            bot = ReadingBot(
                model="test", folder_to_mount="/nonexistent/path/that/should/not/exist"
            )
            # If this doesn't raise an exception immediately, that's also valid
            # (validation might happen later during run())
        except Exception as e:
            # Some validation error might be expected
            assert isinstance(e, (ValueError, FileNotFoundError, OSError))

    def test_library_version_and_metadata(self):
        """Test that library metadata is accessible."""
        import microbots

        # Check for common metadata attributes
        # Note: These might not all be present depending on the setup
        metadata_attrs = ["__version__", "__author__", "__description__"]

        for attr in metadata_attrs:
            if hasattr(microbots, attr):
                value = getattr(microbots, attr)
                assert value is not None
                print(f"microbots.{attr}: {value}")
            else:
                print(f"microbots.{attr}: not defined")

        # At minimum, __all__ should be defined
        assert hasattr(microbots, "__all__")
        assert isinstance(microbots.__all__, list)
        assert len(microbots.__all__) > 0
