#!/usr/bin/env python3
"""
Simple test for file copy functionality
"""

import os
import sys
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
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
            # Get path to countries.txt file specifically
            countries_file_path = Path(__file__).parent.parent.parent / "bot" / "countries_to_capital" / "countries_dir" / "countries.txt"
            result = env.copy_to_container(str(countries_file_path), "/var/log/")
            
            # Verify
            print(f"Copy result: {result}")
            if result:
                print("✅ Copy succeeded")
            else:
                print("❌ Copy failed")
            
            # Test copying from container to host
            # Use /tmp/ which is available and writable on all systems
            result_back = env.copy_from_container("/var/log/countries.txt", "/tmp/")
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