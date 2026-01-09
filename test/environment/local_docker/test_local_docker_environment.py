"""
Integration tests for LocalDockerEnvironment
"""
import pytest
import os
import socket
import re
import logging
import time

# Add src to path for imports
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from microbots.environment.local_docker.LocalDockerEnvironment import LocalDockerEnvironment
from microbots.extras.mount import Mount
from microbots.constants import DOCKER_WORKING_DIR

logger = logging.getLogger(__name__)

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

    @pytest.fixture(scope="class")
    def shared_env(self, available_port):
        """Create a single LocalDockerEnvironment instance for all tests in this class"""
        env = None
        try:
            env = LocalDockerEnvironment(port=available_port)

            # Wait for container to be ready
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

        logger.info("Testing heredoc command execution...")
        start_time = time.time()

        # Execute the heredoc command
        result = shared_env.execute(heredoc_command, timeout=60)
        end_time = time.time()
        execution_time = end_time - start_time

        logger.info(f"Heredoc command completed in {execution_time:.2f} seconds")
        logger.info(f"Return code: {result.return_code}")
        logger.info(f"Stdout: {result.stdout}")
        logger.info(f"Stderr: {result.stderr}")

        # The command should complete successfully (converted automatically)
        assert result.return_code == 0, f"Heredoc command failed with return code {result.return_code}"

        # Should complete in reasonable time (less than 30 seconds)
        assert execution_time < 30, f"Heredoc command took too long: {execution_time:.2f} seconds"

        # Verify the file was created correctly
        verify_result = shared_env.execute("cat /tmp/test_heredoc.py")
        assert verify_result.return_code == 0
        assert "missing_colon_error" in verify_result.stdout
        assert re.search(r"if True$", verify_result.stdout, re.MULTILINE) is not None  # Check for "if True" at end of line (missing colon)
        logger.info(f"Verify result: {verify_result}")

        # Test that the Python file has the expected syntax error
        python_result = shared_env.execute("python3 /tmp/test_heredoc.py")
        # Should fail due to syntax error (missing colon)
        assert python_result.return_code != 0
        assert "SyntaxError" in python_result.stderr or "invalid syntax" in python_result.stderr

        logger.info("Heredoc command with automatic conversion test passed successfully")

    @pytest.mark.integration
    @pytest.mark.docker
    @pytest.mark.slow
    def test_read_write_mount(self, test_repo):
        """Test READ_WRITE mount functionality - creates own env because mounting requires initialization-time config"""
        # Get a fresh port for this test since shared_env is using the class-scoped port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            mount_port = s.getsockname()[1]

        env = None
        test_repo_mount = Mount(
            host_path=str(test_repo),
            sandbox_path=f"{DOCKER_WORKING_DIR}/{test_repo.name}",
            permission="READ_WRITE"
        )
        try:
            env = LocalDockerEnvironment(
                port=mount_port,
                folder_to_mount=test_repo_mount,
            )

            mount_path = f"{DOCKER_WORKING_DIR}/{test_repo.name}"

            # Test that mounted directory is accessible
            result = env.execute(f"ls {mount_path}")
            assert result.return_code == 0
            assert "tests" in result.stdout

            # Test reading subdirectory
            result = env.execute(f"ls {mount_path}/tests")
            assert result.return_code == 0
            assert "missing_colon.py" in result.stdout

            # Test writing to the mounted directory (should succeed with READ_WRITE)
            result = env.execute(f"echo 'new content from container' > {mount_path}/new_test_file.txt")
            assert result.return_code == 0

            # Verify the file was created on the host
            new_file_path = test_repo / "new_test_file.txt"
            assert new_file_path.exists()
            with open(new_file_path, 'r') as f:
                content = f.read().strip()
                assert "new content from container" in content

            # Clean up the created file
            new_file_path.unlink()

        finally:
            if env:
                env.stop()

    @pytest.mark.integration
    @pytest.mark.docker
    @pytest.mark.slow
    def test_read_only_mount(self, test_repo):
        """Test READ_ONLY mount with overlay functionality - creates own env because mounting requires initialization-time config"""
        # Get a fresh port for this test since shared_env is using the class-scoped port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            mount_port = s.getsockname()[1]

        env = None
        test_repo_mount = Mount(
            host_path=str(test_repo),
            sandbox_path=f"{DOCKER_WORKING_DIR}/{test_repo.name}",
            permission="READ_ONLY"
        )

        try:
            env = LocalDockerEnvironment(
                port=mount_port,
                folder_to_mount=test_repo_mount,
            )

            mount_path = f"{DOCKER_WORKING_DIR}/{test_repo.name}"

            # Test that mounted directory is accessible
            result = env.execute(f"ls {mount_path}")
            assert result.return_code == 0
            assert "tests" in result.stdout

            # Test reading subdirectory
            result = env.execute(f"ls {mount_path}/tests")
            assert result.return_code == 0
            assert "missing_colon.py" in result.stdout

            # Test writing to the mounted directory (should appear to succeed with overlay)
            result = env.execute(f"echo 'overlay content' > {mount_path}/overlay_file.txt")
            assert result.return_code == 0

            # Verify the file appears to exist in container
            result = env.execute(f"cat {mount_path}/overlay_file.txt")
            assert result.return_code == 0
            assert "overlay content" in result.stdout

            # Verify the file was NOT created on the host (read-only mount)
            overlay_file_path = test_repo / "overlay_file.txt"
            assert not overlay_file_path.exists()

            # Test modifying existing file (should work in overlay)
            result = env.execute(f"echo 'overlay modification' >> {mount_path}/tests/missing_colon.py")
            assert result.return_code == 0

            # Verify original file on host is unchanged
            with open(test_repo / "tests" / "missing_colon.py", 'r') as f:
                content = f.read()
                assert "overlay modification" not in content

        finally:
            if env:
                env.stop()

    @pytest.mark.integration
    @pytest.mark.docker
    def test_copy_to_container(self, shared_env, test_repo):
        """Test copying files from host to container using shared environment"""
        # Test copying a single file
        source_file = test_repo / "tests" / "missing_colon.py"
        dest_dir = "/tmp/"

        success = shared_env.copy_to_container(str(source_file), dest_dir)
        assert success is True

        # Verify file exists and has correct content in container
        result = shared_env.execute(f"cat {dest_dir}missing_colon.py")
        assert result.return_code == 0
        assert "division" in result.stdout

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

    @pytest.mark.integration
    @pytest.mark.docker
    def test_command_timeout(self, shared_env):
        """Test that commands timeout correctly and return appropriate error code"""
        # Execute a command that should timeout
        start_time = time.time()
        result = shared_env.execute("sleep 10", timeout=2)
        elapsed = time.time() - start_time

        # Should timeout and return exit code 124
        assert result.return_code == 124, f"Expected timeout exit code 124, got {result.return_code}"

        # Should contain timeout message in stderr
        assert "timeout" in result.stderr.lower() or "timed out" in result.stderr.lower(), \
            f"Expected timeout message in stderr, got: {result.stderr}"

        # Should timeout quickly - around 2s for command + 5s for recovery attempt
        assert elapsed < 10, f"Command should timeout in ~2-7s (including recovery), took {elapsed:.1f}s"

    @pytest.mark.integration
    @pytest.mark.docker
    def test_shell_recovery_after_timeout(self, shared_env):
        """Test that shell remains responsive after a command timeout"""

        # First, execute a command that will timeout
        logger.info("1. Executing command that will timeout...")
        result = shared_env.execute("sleep 10", timeout=2)
        assert result.return_code == 124, "Command should timeout with exit code 124"
        logger.info("   ✓ Command timed out as expected")

        # Give recovery mechanism more time to work (shell recovery + HTTP recovery)
        time.sleep(3)

        # Now try to execute a normal command - shell should be responsive
        logger.info("2. Testing if shell is still responsive after timeout...")
        result = shared_env.execute("echo 'Shell is responsive'", timeout=10)

        assert result.return_code == 0, \
            f"Shell should be responsive after timeout. Got return code {result.return_code}, stderr: {result.stderr}"
        assert "Shell is responsive" in result.stdout, \
            f"Output should contain expected text. Got: {result.stdout}"
        logger.info("   ✓ Shell recovered successfully")

        # Execute multiple commands to ensure stability
        logger.info("3. Testing multiple commands after recovery...")
        for i in range(3):
            result = shared_env.execute(f"echo 'Command {i+1}'", timeout=5)
            assert result.return_code == 0, f"Command {i+1} should succeed after recovery"
            assert f"Command {i+1}" in result.stdout, f"Command {i+1} output should be correct"
            logger.info(f"   ✓ Command {i+1} successful")

    @pytest.mark.integration
    @pytest.mark.docker
    def test_multiple_timeouts(self, shared_env):
        """Test that shell can handle multiple consecutive timeouts"""
        for i in range(3):
            logger.info(f"Timeout test {i+1}/3...")

            # Execute a command that will timeout
            result = shared_env.execute("sleep 5", timeout=1)
            assert result.return_code == 124, f"Timeout {i+1} should return exit code 124"

            # Wait a moment for recovery
            time.sleep(0.5)

            # Verify shell is still responsive
            result = shared_env.execute(f"echo 'After timeout {i+1}'", timeout=5)
            assert result.return_code == 0, f"Shell should be responsive after timeout {i+1}"
            assert f"After timeout {i+1}" in result.stdout, f"Output {i+1} should be correct"
            logger.info(f"   ✓ Recovery {i+1} successful")
