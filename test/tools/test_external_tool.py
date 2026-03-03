"""
Unit tests for ExternalTool.

Every public and private method is exercised — both success and failure paths —
to achieve 100 % line/branch coverage of ``external_tool.py``.
"""

import os
import stat
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

from microbots.tools.external_tool import ExternalTool
from microbots.tools.tool import TOOLTYPE, EnvFileCopies


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool(**overrides):
    """Create a concrete ExternalTool with sensible defaults.

    ExternalTool is a pydantic dataclass so we must supply every required
    field that ToolAbstract declares.
    """
    defaults = dict(
        name="test-tool",
        description="A tool used in tests",
        usage_instructions_to_llm="Use it wisely",
        install_commands=[],
        env_variables=[],
        verify_commands=[],
        setup_commands=[],
        uninstall_commands=[],
        files_to_copy=[],
    )
    defaults.update(overrides)
    return ExternalTool(**defaults)


def _ok_result(stdout="", stderr=""):
    """Simulate a successful subprocess.CompletedProcess."""
    return subprocess.CompletedProcess(
        args="", returncode=0, stdout=stdout, stderr=stderr
    )


def _fail_result(stdout="", stderr="oops"):
    """Simulate a failed subprocess.CompletedProcess."""
    return subprocess.CompletedProcess(
        args="", returncode=1, stdout=stdout, stderr=stderr
    )


# Shared mock environment — none of the ExternalTool methods actually use it,
# but the abstract interface requires it.
_mock_env = MagicMock()


# ---------------------------------------------------------------------------
# tool_type
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExternalToolType:

    def test_tool_type_is_external(self):
        tool = _make_tool()
        assert tool.tool_type == TOOLTYPE.EXTERNAL


# ---------------------------------------------------------------------------
# _run_host_command
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRunHostCommand:

    def test_runs_command_via_subprocess(self):
        with patch("microbots.tools.external_tool.subprocess.run",
                    return_value=_ok_result(stdout="hello")) as mock_run:
            result = ExternalTool._run_host_command("echo hello")

        mock_run.assert_called_once_with(
            "echo hello", shell=True, capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert result.stdout == "hello"


# ---------------------------------------------------------------------------
# _verify_env_variables
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVerifyEnvVariables:

    def test_no_env_variables_succeeds(self):
        """Empty env_variables list should not raise."""
        tool = _make_tool(env_variables=[])
        tool._verify_env_variables()  # no exception

    def test_present_variables_succeeds(self):
        tool = _make_tool(env_variables=["PATH"])
        tool._verify_env_variables()  # PATH is always set

    def test_missing_variable_raises(self):
        tool = _make_tool(env_variables=["NONEXISTENT_VAR_12345"])
        with pytest.raises(EnvironmentError, match="Missing required environment variable"):
            tool._verify_env_variables()

    def test_multiple_missing_variables_listed(self):
        tool = _make_tool(env_variables=["MISSING_A_999", "MISSING_B_999"])
        with pytest.raises(EnvironmentError, match="MISSING_A_999.*MISSING_B_999"):
            tool._verify_env_variables()

    def test_none_env_variables_succeeds(self):
        """When env_variables is None, treat as empty."""
        tool = _make_tool(env_variables=None)
        tool._verify_env_variables()


# ---------------------------------------------------------------------------
# _copy_files
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCopyFiles:

    def test_empty_files_list(self):
        tool = _make_tool(files_to_copy=[])
        tool._copy_files()  # nothing to copy, should succeed silently

    def test_none_files_list(self):
        tool = _make_tool(files_to_copy=None)
        tool._copy_files()

    def test_copy_happy_path(self, tmp_path):
        """Source exists → copied to dest with correct permissions."""
        src = tmp_path / "source.txt"
        src.write_text("data")
        dest = tmp_path / "out" / "dest.txt"

        fc = EnvFileCopies(src=src, dest=dest, permissions=7)
        tool = _make_tool(files_to_copy=[fc])
        tool._copy_files()

        assert dest.exists()
        assert dest.read_text() == "data"
        # permissions 7 → octal (7<<6)|(7<<3) = 0o770
        mode = dest.stat().st_mode
        assert mode & 0o770 == 0o770

    def test_copy_source_not_found_raises(self, tmp_path):
        src = tmp_path / "nonexistent.txt"
        dest = tmp_path / "dest.txt"
        fc = EnvFileCopies(src=src, dest=dest, permissions=4)
        tool = _make_tool(files_to_copy=[fc])

        with pytest.raises(ValueError, match="not found on host"):
            tool._copy_files()

    def test_copy_invalid_permissions_raises(self, tmp_path):
        """Permissions outside 0-7 should raise ValueError."""
        src = tmp_path / "src.txt"
        src.write_text("x")
        dest = tmp_path / "dst.txt"

        # EnvFileCopies.__post_init__ validates permissions 0-7, so we need
        # to bypass by setting it after construction.
        fc = EnvFileCopies(src=src, dest=dest, permissions=5)
        # Force an out-of-range value after pydantic validation
        object.__setattr__(fc, "permissions", 9)
        tool = _make_tool(files_to_copy=[fc])

        with pytest.raises(ValueError, match="Invalid permissions"):
            tool._copy_files()

    def test_copy_creates_parent_dirs(self, tmp_path):
        """Dest parent dir should be created automatically."""
        src = tmp_path / "f.txt"
        src.write_text("hi")
        dest = tmp_path / "a" / "b" / "c" / "f.txt"

        fc = EnvFileCopies(src=src, dest=dest, permissions=6)
        tool = _make_tool(files_to_copy=[fc])
        tool._copy_files()

        assert dest.exists()


# ---------------------------------------------------------------------------
# install_tool
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestInstallTool:

    def test_no_commands(self):
        tool = _make_tool(install_commands=[])
        with patch.object(tool, "_copy_files") as mock_cp:
            tool.install_tool(_mock_env)
        mock_cp.assert_called_once()

    def test_success_commands(self):
        tool = _make_tool(install_commands=["cmd1", "cmd2"])
        with patch.object(ExternalTool, "_run_host_command",
                          return_value=_ok_result()) as mock_run, \
             patch.object(tool, "_copy_files"):
            tool.install_tool(_mock_env)

        assert mock_run.call_count == 2
        mock_run.assert_any_call("cmd1")
        mock_run.assert_any_call("cmd2")

    def test_failing_command_raises(self):
        tool = _make_tool(install_commands=["bad_cmd"])
        with patch.object(ExternalTool, "_run_host_command",
                          return_value=_fail_result(stderr="not found")):
            with pytest.raises(RuntimeError, match="Failed to install external tool"):
                tool.install_tool(_mock_env)


# ---------------------------------------------------------------------------
# verify_tool_installation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVerifyToolInstallation:

    def test_no_commands(self):
        tool = _make_tool(verify_commands=[])
        tool.verify_tool_installation(_mock_env)  # should not raise

    def test_success_commands(self):
        tool = _make_tool(verify_commands=["check1"])
        with patch.object(ExternalTool, "_run_host_command",
                          return_value=_ok_result()):
            tool.verify_tool_installation(_mock_env)

    def test_failing_command_raises(self):
        tool = _make_tool(verify_commands=["check_fail"])
        with patch.object(ExternalTool, "_run_host_command",
                          return_value=_fail_result(stderr="missing lib")):
            with pytest.raises(RuntimeError, match="Failed to verify external tool"):
                tool.verify_tool_installation(_mock_env)


# ---------------------------------------------------------------------------
# setup_tool
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSetupTool:

    def test_no_commands_no_env(self):
        tool = _make_tool(setup_commands=[], env_variables=[])
        tool.setup_tool(_mock_env)

    def test_env_verified_before_setup_commands(self):
        """_verify_env_variables should be called before setup commands run."""
        call_order = []
        tool = _make_tool(
            setup_commands=["setup1"],
            env_variables=["PATH"],
        )

        original_verify = tool._verify_env_variables

        def track_verify():
            call_order.append("verify")
            original_verify()

        with patch.object(tool, "_verify_env_variables", side_effect=track_verify), \
             patch.object(ExternalTool, "_run_host_command",
                          side_effect=lambda cmd: (call_order.append("run"), _ok_result())[1]):
            tool.setup_tool(_mock_env)

        assert call_order == ["verify", "run"]

    def test_success_commands(self):
        tool = _make_tool(setup_commands=["setup_a", "setup_b"])
        with patch.object(ExternalTool, "_run_host_command",
                          return_value=_ok_result()) as mock_run:
            tool.setup_tool(_mock_env)
        assert mock_run.call_count == 2

    def test_failing_command_raises(self):
        tool = _make_tool(setup_commands=["fail_cmd"])
        with patch.object(ExternalTool, "_run_host_command",
                          return_value=_fail_result(stderr="boom")):
            with pytest.raises(RuntimeError, match="Failed to setup external tool"):
                tool.setup_tool(_mock_env)

    def test_missing_env_variable_raises(self):
        tool = _make_tool(
            setup_commands=["anything"],
            env_variables=["VERY_UNLIKELY_VAR_XYZ"],
        )
        with pytest.raises(EnvironmentError, match="Missing required"):
            tool.setup_tool(_mock_env)


# ---------------------------------------------------------------------------
# uninstall_tool
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUninstallTool:

    def test_no_files_no_commands(self):
        tool = _make_tool(files_to_copy=[], uninstall_commands=[])
        tool.uninstall_tool(_mock_env)

    def test_removes_existing_file(self, tmp_path):
        dest = tmp_path / "tool_file.txt"
        dest.write_text("data")
        src = tmp_path / "src.txt"
        src.write_text("data")

        fc = EnvFileCopies(src=src, dest=dest, permissions=6)
        tool = _make_tool(files_to_copy=[fc], uninstall_commands=[])
        tool.uninstall_tool(_mock_env)

        assert not dest.exists()

    def test_missing_file_logs_warning(self, tmp_path):
        dest = tmp_path / "ghost.txt"
        src = tmp_path / "src.txt"
        src.write_text("x")

        fc = EnvFileCopies(src=src, dest=dest, permissions=4)
        tool = _make_tool(files_to_copy=[fc], uninstall_commands=[])
        # Should not raise — just warns
        tool.uninstall_tool(_mock_env)

    def test_success_commands(self):
        tool = _make_tool(uninstall_commands=["cleanup"])
        with patch.object(ExternalTool, "_run_host_command",
                          return_value=_ok_result()):
            tool.uninstall_tool(_mock_env)

    def test_failing_command_raises(self):
        tool = _make_tool(uninstall_commands=["bad_cleanup"])
        with patch.object(ExternalTool, "_run_host_command",
                          return_value=_fail_result(stderr="denied")):
            with pytest.raises(RuntimeError, match="Failed to uninstall external tool"):
                tool.uninstall_tool(_mock_env)
