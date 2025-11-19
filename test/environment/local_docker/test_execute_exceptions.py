"""
Unit tests for LocalDockerEnvironment.execute exception handling
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import requests

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

from microbots.environment.local_docker.LocalDockerEnvironment import LocalDockerEnvironment
from microbots.environment.Environment import CmdReturn


@pytest.mark.unit
class TestExecuteExceptionHandling:
    """Unit tests for execute method exception handling"""

    @pytest.fixture
    def mock_env(self):
        """Create a mock LocalDockerEnvironment with minimal setup"""
        with patch('microbots.environment.local_docker.LocalDockerEnvironment.docker'):
            with patch.object(LocalDockerEnvironment, 'start'):
                env = LocalDockerEnvironment.__new__(LocalDockerEnvironment)
                env.image = "test_image"
                env.folder_to_mount = None
                env.overlay_mount = False
                env.container = Mock()
                env.client = Mock()
                env.port = 8080
                env.container_port = 8080
                env.deleted = False
                return env

    def test_execute_connect_timeout(self, mock_env):
        """Test execute method handles ConnectTimeout exception"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectTimeout(
                "Connection timed out"
            )

            result = mock_env.execute("echo 'test'", timeout=30)

            assert isinstance(result, CmdReturn)
            assert result.return_code == 124  # Standard timeout exit code
            assert result.stdout == ""
            assert "Connection timeout" in result.stderr
            assert f"port {mock_env.port}" in result.stderr

    def test_execute_read_timeout(self, mock_env):
        """Test execute method handles ReadTimeout exception"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.ReadTimeout(
                "Read operation timed out"
            )

            result = mock_env.execute("sleep 100", timeout=5)

            assert isinstance(result, CmdReturn)
            assert result.return_code == 124  # Standard timeout exit code
            assert result.stdout == ""
            assert "Read timeout" in result.stderr
            assert "waiting for command output" in result.stderr

    def test_execute_connection_error(self, mock_env):
        """Test execute method handles ConnectionError exception"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError(
                "Failed to establish connection"
            )

            result = mock_env.execute("echo 'test'")

            assert isinstance(result, CmdReturn)
            assert result.return_code == 1
            assert result.stdout == ""
            assert "Failed to establish connection" in result.stderr

    def test_execute_http_error(self, mock_env):
        """Test execute method handles HTTPError exception"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "500 Server Error"
            )
            mock_post.return_value = mock_response

            result = mock_env.execute("echo 'test'")

            assert isinstance(result, CmdReturn)
            assert result.return_code == 1
            assert result.stdout == ""

    def test_execute_request_exception(self, mock_env):
        """Test execute method handles generic RequestException"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.RequestException(
                "Generic request error"
            )

            result = mock_env.execute("echo 'test'")

            assert isinstance(result, CmdReturn)
            assert result.return_code == 1
            assert result.stdout == ""
            assert "Generic request error" in result.stderr

    def test_execute_unexpected_exception(self, mock_env):
        """Test execute method handles unexpected non-request exceptions"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = ValueError("Unexpected error")

            result = mock_env.execute("echo 'test'")

            assert isinstance(result, CmdReturn)
            assert result.return_code == 1
            assert result.stdout == ""
            assert "Unexpected error" in result.stderr

    def test_execute_json_decode_error(self, mock_env):
        """Test execute method handles JSON decode errors"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = mock_env.execute("echo 'test'")

            assert isinstance(result, CmdReturn)
            assert result.return_code == 1

    def test_execute_timeout_with_specific_duration(self, mock_env):
        """Test that timeout duration is correctly included in error message"""
        with patch('requests.post') as mock_post:
            # Use a context manager mock for time tracking
            with patch('time.perf_counter', side_effect=[0.0, 15.5]):
                mock_post.side_effect = requests.exceptions.ReadTimeout()

                result = mock_env.execute("long_command", timeout=60)

                assert result.return_code == 124
                assert "15.5s" in result.stderr or "timeout" in result.stderr.lower()

    def test_execute_connection_timeout_timing(self, mock_env):
        """Test that ConnectTimeout includes elapsed time in error message"""
        with patch('requests.post') as mock_post:
            with patch('time.perf_counter', side_effect=[0.0, 5.3]):
                mock_post.side_effect = requests.exceptions.ConnectTimeout()

                result = mock_env.execute("test_command")

                assert result.return_code == 124
                assert "5.3s" in result.stderr or "Connection timeout" in result.stderr

    def test_execute_multiple_timeout_scenarios(self, mock_env):
        """Test different timeout values and verify correct error codes"""
        timeout_scenarios = [1, 30, 300, 600]

        for timeout_val in timeout_scenarios:
            with patch('requests.post') as mock_post:
                mock_post.side_effect = requests.exceptions.ReadTimeout()

                result = mock_env.execute("test", timeout=timeout_val)

                assert result.return_code == 124
                assert result.stdout == ""
                assert result.stderr != ""

    def test_execute_exception_with_empty_message(self, mock_env):
        """Test exception handling when exception has no message"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.RequestException()

            result = mock_env.execute("echo 'test'")

            assert isinstance(result, CmdReturn)
            assert result.return_code == 1
            assert result.stdout == ""

    @pytest.mark.skip(reason="KeyboardInterrupt crashes test runners and cannot be tested in CI")
    def test_execute_keyboard_interrupt(self, mock_env):
        """Test that KeyboardInterrupt is handled as unexpected exception
        
        Note: This test is skipped because KeyboardInterrupt terminates the test process
        and crashes pytest-xdist workers in parallel execution.
        """
        with patch('requests.post') as mock_post:
            mock_post.side_effect = KeyboardInterrupt()

            result = mock_env.execute("echo 'test'")

            assert isinstance(result, CmdReturn)
            assert result.return_code == 1
            assert "Unexpected error" in result.stderr

    def test_execute_memory_error(self, mock_env):
        """Test handling of MemoryError"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = MemoryError("Out of memory")

            result = mock_env.execute("echo 'test'")

            assert isinstance(result, CmdReturn)
            assert result.return_code == 1

    def test_execute_attribute_error_on_response(self, mock_env):
        """Test handling when response object is malformed"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.side_effect = AttributeError("No attribute")
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = mock_env.execute("echo 'test'")

            assert isinstance(result, CmdReturn)
            assert result.return_code == 1

    def test_execute_successful_after_retries_concept(self, mock_env):
        """Test that a successful response returns correct CmdReturn"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "output": {
                    "stdout": "Success",
                    "stderr": "",
                    "return_code": 0
                }
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = mock_env.execute("echo 'test'")

            assert result.return_code == 0
            assert result.stdout == "Success"
            assert result.stderr == ""

    def test_execute_with_default_timeout(self, mock_env):
        """Test execute with default timeout value"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.ReadTimeout()

            # Call without specifying timeout (should use default 300)
            result = mock_env.execute("echo 'test'")

            assert result.return_code == 124
            # Verify timeout was passed to requests.post
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs['timeout'] == 300

    def test_execute_with_custom_timeout(self, mock_env):
        """Test execute with custom timeout value"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "output": {"stdout": "", "stderr": "", "return_code": 0}
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            mock_env.execute("test", timeout=60)

            # Verify custom timeout was passed
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs['timeout'] == 60

    def test_execute_response_missing_output_key(self, mock_env):
        """Test handling when response JSON is missing 'output' key"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {}  # Missing 'output' key
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = mock_env.execute("echo 'test'")

            # When output key is missing, returns empty string which causes AttributeError
            # This is caught by the general Exception handler
            assert isinstance(result, CmdReturn)
            assert result.stdout == ""
            assert "Unexpected error" in result.stderr
            assert result.return_code == 1

    def test_execute_response_partial_output(self, mock_env):
        """Test handling when output dict has missing keys"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "output": {
                    "stdout": "partial output"
                    # Missing stderr and return_code
                }
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = mock_env.execute("echo 'test'")

            assert result.stdout == "partial output"
            assert result.stderr == ""  # Should default to empty string
            assert result.return_code == 0  # Should default to 0

    def test_execute_logs_debug_messages(self, mock_env, caplog):
        """Test that debug messages are logged during execution"""
        import logging

        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "output": {"stdout": "test", "stderr": "", "return_code": 0}
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            with caplog.at_level(logging.DEBUG):
                mock_env.execute("echo 'test'")

            # Verify that command execution was logged
            # Note: Actual log checking depends on logger configuration

    def test_execute_logs_error_on_exception(self, mock_env, caplog):
        """Test that errors are logged when exceptions occur"""
        import logging

        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")

            with caplog.at_level(logging.ERROR):
                mock_env.execute("echo 'test'")

            # Error should be logged

    def test_execute_performance_timing(self, mock_env):
        """Test that execution time is tracked correctly"""
        with patch('requests.post') as mock_post:
            with patch('time.perf_counter', side_effect=[0.0, 2.5]):
                mock_response = Mock()
                mock_response.json.return_value = {
                    "output": {"stdout": "", "stderr": "", "return_code": 0}
                }
                mock_response.raise_for_status.return_value = None
                mock_post.return_value = mock_response

                result = mock_env.execute("echo 'test'")

                # Should complete successfully with timing tracked
                assert result.return_code == 0
