#!/usr/bin/env python3
"""
Simple test for file copy functionality
"""

import os
import sys
from pathlib import Path

# Add src directory to path to import from local source
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)

from microbots.environment.local_docker import LocalDockerEnvironment
from microbots.utils.path import get_file_mount_info


class TestFileCopy:
    """Simple test for file copy"""

    def test_copy_file(self):
        """Test copying a file to container and from container to host"""
        # Create environment with a different port to avoid conflicts
        env = LocalDockerEnvironment(port=8083)

        try:
            # Copy to container

            file_mount_result = get_file_mount_info(
                f"{str(Path(__file__).parent)}/LocalDockerEnvironmentTest.py"
            )
            print(file_mount_result, "file_mount_result==================")

            # Give absolute path
            result = env.copy_to_container(
                f"{str(Path(__file__).parent)}/LocalDockerEnvironmentTest.py",
                "/var/log/LocalDockerEnvironmentTest.py",
            )

            # Verify
            print(f"Copy result: {result}")
            if result:
                print("✅ Copy succeeded")
            else:
                print("❌ Copy failed")

            # env.stop()

            # # Test copying from container to host
            # result_back = env.copy_from_container("/var/log/README.md", "/home/kkaitepalli/temp/README_copied.md")
            # print(f"Copy back result: {result_back}")
            # if result_back:
            #     print("✅ Copy back succeeded")
            # else:
            #     print("❌ Copy back failed")

        finally:
            # Cleanup
            # os.unlink(test_file)
            # env.stop()
            print("Not stopping environment for debug")


if __name__ == "__main__":
    # Run the tests
    test_instance = TestFileCopy()
    test_instance.test_copy_file()
