"""
Integration tests for LocalDockerEnvironment
"""
import pytest
import shutil
import os
import time
import socket
from pathlib import Path
import re

# Add src to path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from microbots.environment.local_docker.LocalDockerEnvironment import LocalDockerEnvironment
from microbots.environment.Environment import CmdReturn

# Use the correct working directory path
DOCKER_WORKING_DIR = "workdir"

class TestLocalDockerEnvironmentIntegration:
    """Integration tests for LocalDockerEnvironment with real Docker containers"""

    @pytest.fixture(scope="class")
    def available_port(self):
        """Find an available port for testing - class scoped to reuse same port"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    @pytest.fixture
    def test_dir(self):
        """Use existing test/bot/calculator directory for testing instead of temp directory"""
        # Get the base path of the minions project dynamically
        current_file_dir = os.path.dirname(os.path.abspath(__file__))  # /path/to/minions/test/environment/local_docker
        minions_base_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file_dir)))  # /path/to/minions
        test_directory = os.path.join(minions_base_path, "test", "bot", "calculator")

        # Verify the directory exists and has test files
        assert os.path.exists(test_directory), f"Test directory {test_directory} does not exist"
        assert os.path.exists(os.path.join(test_directory, "calculator.log")), "Test file calculator.log not found"
        assert os.path.exists(os.path.join(test_directory, "code")), "Test code subdirectory not found"
        assert os.path.exists(os.path.join(test_directory, "code", "calculator.py")), "Test file calculator.py not found"

        return test_directory

    @pytest.fixture(scope="class")
    def shared_env(self, available_port):
        """Create a single LocalDockerEnvironment instance for all tests in this class"""
        env = None
        try:
            env = LocalDockerEnvironment(port=available_port)

            # Wait for container to be ready
            import time
            time.sleep(2)

            # Verify it's working
            result = env.execute("echo 'Environment ready'")
            assert result.return_code == 0

            yield env
        finally:
            if env:
                env.stop()

    @pytest.mark.integration
    @pytest.mark.docker
    def test_basic_environment_lifecycle(self, shared_env):
        """Test basic environment functionality using shared environment"""
        # Test that container is running
        assert shared_env.container is not None
        shared_env.container.reload()
        assert shared_env.container.status == 'running'

        # Test that we can connect and execute commands
        result = shared_env.execute("echo 'Hello World'")
        assert result.return_code == 0
        assert "Hello World" in result.stdout


    @pytest.mark.integration
    @pytest.mark.docker
    def test_initialization_validation_errors(self):
        """Test that initialization properly validates parameters"""
        # Test permission without folder
        with pytest.raises(ValueError, match="permission provided but folder_to_mount is None"):
            LocalDockerEnvironment(port=8999, permission="READ_ONLY")

        # Test folder without permission
        with pytest.raises(ValueError, match="folder_to_mount provided but permission is None"):
            LocalDockerEnvironment(port=8999, folder_to_mount="/some/path")

        # Test invalid permission
        with pytest.raises(ValueError, match="permission must be 'READ_ONLY' or 'READ_WRITE' when provided"):
            LocalDockerEnvironment(port=8999, folder_to_mount="/some/path", permission="INVALID")

    @pytest.mark.integration
    @pytest.mark.docker
    def test_command_execution_basic(self, shared_env):
        """Test basic command execution functionality using shared environment"""
        # Test simple echo
        result = shared_env.execute("echo 'test message'")
        assert result.return_code == 0
        assert "test message" in result.stdout
        assert result.stderr == ""

        # Test command with error
        result = shared_env.execute("nonexistent_command")
        assert result.return_code != 0
        assert result.stderr != ""

        # Test pwd
        result = shared_env.execute("pwd")
        assert result.return_code == 0
        assert "/" in result.stdout

    @pytest.mark.integration
    @pytest.mark.docker
    def test_command_execution_complex(self, shared_env):
        """Test that heredoc commands are automatically converted to safe alternatives"""
        import time

        # Test the specific heredoc command that was causing timeouts
        heredoc_command = """cat > /tmp/test_heredoc.py << EOF
#!/usr/bin/env python3
import sys

def missing_colon_error():
    # This function demonstrates a syntax error - missing colon after if statement
    if True
        print("This will cause a syntax error")
        return True

    return False

if __name__ == "__main__":
    try:
        result = missing_colon_error()
        print(f"Function result: {result}")
    except SyntaxError as e:
        print(f"Syntax error caught: {e}")
        sys.exit(1)
EOF"""

        print("Testing heredoc command execution...")
        start_time = time.time()

        # Execute the heredoc command
        result = shared_env.execute(heredoc_command, timeout=60)
        end_time = time.time()
        execution_time = end_time - start_time

        print(f"Heredoc command completed in {execution_time:.2f} seconds")
        print(f"Return code: {result.return_code}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")

        # The command should complete successfully (converted automatically)
        assert result.return_code == 0, f"Heredoc command failed with return code {result.return_code}"

        # Should complete in reasonable time (less than 30 seconds)
        assert execution_time < 30, f"Heredoc command took too long: {execution_time:.2f} seconds"

        # Verify the file was created correctly
        verify_result = shared_env.execute("cat /tmp/test_heredoc.py")
        assert verify_result.return_code == 0
        assert "missing_colon_error" in verify_result.stdout
        assert re.search(r"if True$", verify_result.stdout, re.MULTILINE) is not None  # Check for "if True" at end of line (missing colon)
        print(verify_result)

        # Test that the Python file has the expected syntax error
        python_result = shared_env.execute("python3 /tmp/test_heredoc.py")
        # Should fail due to syntax error (missing colon)
        assert python_result.return_code != 0
        assert "SyntaxError" in python_result.stderr or "invalid syntax" in python_result.stderr

        print("Heredoc command with automatic conversion test passed successfully")

    @pytest.mark.integration
    @pytest.mark.docker
    def test_read_write_mount(self, test_dir):
        """Test READ_WRITE mount functionality - creates own env because mounting requires initialization-time config"""
        # Get a fresh port for this test since shared_env is using the class-scoped port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            mount_port = s.getsockname()[1]

        env = None
        try:
            env = LocalDockerEnvironment(
                port=mount_port,
                folder_to_mount=test_dir,
                permission="READ_WRITE"
            )

            folder_name = os.path.basename(test_dir)
            mount_path = f"/{DOCKER_WORKING_DIR}/{folder_name}"

            # Test that mounted directory is accessible
            result = env.execute(f"ls {mount_path}")
            assert result.return_code == 0
            assert "calculator.log" in result.stdout
            assert "code" in result.stdout

            # Test reading the mounted file
            result = env.execute(f"cat {mount_path}/calculator.log")
            assert result.return_code == 0
            assert "Calculator application started" in result.stdout

            # Test reading subdirectory
            result = env.execute(f"ls {mount_path}/code")
            assert result.return_code == 0
            assert "calculator.py" in result.stdout

            # Test writing to the mounted directory (should succeed with READ_WRITE)
            result = env.execute(f"echo 'new content from container' > {mount_path}/new_test_file.txt")
            assert result.return_code == 0

            # Verify the file was created on the host
            new_file_path = os.path.join(test_dir, "new_test_file.txt")
            assert os.path.exists(new_file_path)
            with open(new_file_path, 'r') as f:
                content = f.read().strip()
                assert "new content from container" in content

            # Clean up the created file
            os.remove(new_file_path)

        finally:
            if env:
                env.stop()

    @pytest.mark.integration
    @pytest.mark.docker
    def test_read_only_mount(self, test_dir):
        """Test READ_ONLY mount with overlay functionality - creates own env because mounting requires initialization-time config"""
        # Get a fresh port for this test since shared_env is using the class-scoped port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            mount_port = s.getsockname()[1]

        env = None
        try:
            env = LocalDockerEnvironment(
                port=mount_port,
                folder_to_mount=test_dir,
                permission="READ_ONLY"
            )

            folder_name = os.path.basename(test_dir)
            mount_path = f"/{DOCKER_WORKING_DIR}/{folder_name}"

            # Test that mounted directory is accessible
            result = env.execute(f"ls {mount_path}")
            assert result.return_code == 0
            assert "calculator.log" in result.stdout

            # Test reading the mounted file
            result = env.execute(f"cat {mount_path}/calculator.log")
            assert result.return_code == 0
            assert "Calculator application started" in result.stdout

            # Test writing to the mounted directory (should appear to succeed with overlay)
            result = env.execute(f"echo 'overlay content' > {mount_path}/overlay_file.txt")
            assert result.return_code == 0

            # Verify the file appears to exist in container
            result = env.execute(f"cat {mount_path}/overlay_file.txt")
            assert result.return_code == 0
            assert "overlay content" in result.stdout

            # Verify the file was NOT created on the host (read-only mount)
            overlay_file_path = os.path.join(test_dir, "overlay_file.txt")
            assert not os.path.exists(overlay_file_path)

            # Test modifying existing file (should work in overlay)
            result = env.execute(f"echo 'overlay modification' >> {mount_path}/calculator.log")
            assert result.return_code == 0

            # Verify original file on host is unchanged
            with open(os.path.join(test_dir, "calculator.log"), 'r') as f:
                content = f.read()
                assert "overlay modification" not in content
                assert "Calculator application started" in content

        finally:
            if env:
                env.stop()

    @pytest.mark.integration
    @pytest.mark.docker
    def test_copy_to_container(self, shared_env, test_dir):
        """Test copying files from host to container using shared environment"""
        # Test copying a single file
        source_file = os.path.join(test_dir, "calculator.log")
        dest_dir = "/var/log/"

        success = shared_env.copy_to_container(source_file, dest_dir)
        assert success is True

        # Verify file exists and has correct content in container
        result = shared_env.execute(f"cat {dest_dir}calculator.log")
        assert result.return_code == 0
        assert "Calculator application started" in result.stdout

        # Test copying non-existent file
        success = shared_env.copy_to_container("/nonexistent/file.txt", "/tmp/fail.txt")
        assert success is False

    @pytest.mark.integration
    @pytest.mark.docker
    def test_copy_from_container(self, shared_env):
        """Test copying files from container to host using shared environment"""
        # Create a file in container
        container_file = "/tmp/container_created.txt"
        result = shared_env.execute(f"echo 'Created in container' > {container_file}")
        assert result.return_code == 0

        # Copy file from container to host directory
        host_dest_dir = "/tmp/"
        success = shared_env.copy_from_container(container_file, host_dest_dir)
        assert success is True

        # The file should be copied to /tmp/container_created.txt
        copied_file_path = "/tmp/container_created.txt"

        # Verify file exists on host with correct content
        assert os.path.exists(copied_file_path)
        with open(copied_file_path, 'r') as f:
            content = f.read()
            assert "Created in container" in content

        # Clean up the created file
        os.remove(copied_file_path)

        # Test copying non-existent file
        success = shared_env.copy_from_container("/nonexistent/file.txt", "/tmp/fail.txt")
        assert success is False

# Manual test runner function
def run_local_docker_environment_manual_test():
    """Manual test function that can be run outside pytest"""
    print("=== Manual LocalDockerEnvironment Integration Test ===")

    # Get available port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        available_port = s.getsockname()[1]

    try:
        print(f"Creating LocalDockerEnvironment on port {available_port}")
        env = LocalDockerEnvironment(port=available_port)

        print("--- Test 1: Basic Command Execution ---")
        result = env.execute("echo 'Hello from Docker!'")
        print(f"Return Code: {result.return_code}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")

        print("\n--- Test 2: Working Directory ---")
        result = env.execute("pwd")
        print(f"Current directory: {result.stdout.strip()}")

        print("\n--- Test 3: File Operations ---")
        result = env.execute("echo 'test content' > /tmp/test.txt && cat /tmp/test.txt")
        print(f"File operation result: {result.stdout.strip()}")

        print("\n--- Test 4: Container Info ---")
        result = env.execute("hostname && whoami")
        print(f"Container info: {result.stdout.strip()}")

        print("\n--- Manual Test Completed Successfully ---")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.stop()
            print("Environment stopped and cleaned up")


if __name__ == "__main__":
    # Allow running the test file directly for manual testing
    run_local_docker_environment_manual_test()