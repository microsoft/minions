"""
Base test class for integration tests that simulate user experience.

This module provides utilities for testing microbots as users would use them,
with proper imports from the installed package.
"""

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path to simulate installed package
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MicrobotsIntegrationTestBase:
    """Base class for integration tests that simulate user experience."""

    def setup_method(self):
        """Set up test environment before each test."""
        self.test_workspace = None
        self.temp_dirs = []

    def teardown_method(self):
        """Clean up test environment after each test."""
        # Clean up any temporary directories created during tests
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def create_test_workspace(self, name: str = "test_workspace") -> Path:
        """Create a temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp(prefix=f"microbots_test_{name}_")
        self.temp_dirs.append(temp_dir)
        workspace_path = Path(temp_dir)
        self.test_workspace = workspace_path
        return workspace_path

    def create_sample_files(self, workspace: Path, files_content: dict):
        """Create sample files in the workspace for testing.

        Args:
            workspace: Path to the workspace directory
            files_content: Dict mapping file paths to content
        """
        for file_path, content in files_content.items():
            full_path = workspace / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)

    def assert_required_env_vars(self):
        """Assert that required environment variables are set."""
        required_vars = [
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            pytest.skip(f"Missing required environment variables: {missing_vars}")


def requires_azure_openai(func):
    """Decorator to skip tests if Azure OpenAI environment variables are not set."""

    def wrapper(self, *args, **kwargs):
        self.assert_required_env_vars()
        return func(self, *args, **kwargs)

    return wrapper
