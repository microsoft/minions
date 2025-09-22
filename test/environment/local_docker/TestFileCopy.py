#!/usr/bin/env python3
"""
Simple test for file copy functionality
"""

import os
import tempfile
import unittest
import sys

# Add src directory to path to import from local source
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)

from microbots.environment.local_docker import LocalDockerEnvironment

class TestFileCopy():
    """Simple test for file copy"""

    def test_copy_file(self):
        """Test copying a file to container and from container to host"""
        # Create environment
        env = LocalDockerEnvironment(port=8081)
        
        try:
            # Copy to container
            # Give absolute path
            result = env.copy_to_container("/home/kkaitepalli/minions/README.md", "/var/log/README.md")
            
            # Verify
            print(f"Copy result: {result}")
            if result:
                print("✅ Copy succeeded")
            else:
                print("❌ Copy failed")
            
            # Test copying from container to host
            result_back = env.copy_from_container("/var/log/README.md", "/home/kkaitepalli/temp/README_copied.md")
            print(f"Copy back result: {result_back}")
            if result_back:
                print("✅ Copy back succeeded")
            else:
                print("❌ Copy back failed")
            
        finally:
            # Cleanup
            # os.unlink(test_file)
            # env.stop()
            print("Not stopping environment for debug")


if __name__ == "__main__":
    # Run the tests
    test_instance = TestFileCopy()
    test_instance.test_copy_file()