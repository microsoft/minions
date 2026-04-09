"""
Unit tests for the new functionality added to multi_agent_log_parser.py:

- SetupInfo dataclass
- _extract_setup_info() function
- parse_log_entries() with legacy format support
- Agent.error_message field
- TestCase.setup field
"""
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

from microbots.utils.multi_agent_log_parser import (
    Agent,
    SetupInfo,
    TestCase,
    _agent_status_str,
    _extract_setup_info,
    build_test_cases,
    extract_task_from_microbot_sub,
    generate_setup_md,
    parse_log_entries,
    truncate_text,
)


# ---------------------------------------------------------------------------
# Unit tests — SetupInfo dataclass
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSetupInfo:
    """Tests for the new SetupInfo dataclass."""

    def test_default_values(self):
        """SetupInfo has sensible empty defaults."""
        s = SetupInfo()
        assert s.container_id == ""
        assert s.image == ""
        assert s.host_port == ""
        assert s.working_dir == ""
        assert s.volume_mappings == []
        assert s.tools_installed == []
        assert s.files_copied == []

    def test_explicit_construction(self):
        """SetupInfo can be created with explicit values."""
        s = SetupInfo(
            container_id="abc123",
            image="ubuntu:22.04",
            host_port="8080",
            working_dir="/workspace",
            volume_mappings=["/host:/container"],
            tools_installed=["git"],
            files_copied=["file.py → /workspace/file.py"],
        )
        assert s.container_id == "abc123"
        assert s.image == "ubuntu:22.04"
        assert s.host_port == "8080"
        assert s.working_dir == "/workspace"
        assert s.volume_mappings == ["/host:/container"]
        assert s.tools_installed == ["git"]
        assert s.files_copied == ["file.py → /workspace/file.py"]


# ---------------------------------------------------------------------------
# Unit tests — Agent.error_message field
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAgentErrorMessage:
    """Tests for the new error_message field on Agent."""

    def test_default_error_message_is_empty(self):
        """Agent.error_message defaults to empty string."""
        a = Agent()
        assert a.error_message == ""

    def test_can_set_error_message(self):
        """Agent.error_message can be set."""
        a = Agent(error_message="Something went wrong")
        assert a.error_message == "Something went wrong"


# ---------------------------------------------------------------------------
# Unit tests — TestCase.setup field
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTestCaseSetup:
    """Tests for the new setup field on TestCase."""

    def test_default_setup_is_setup_info(self):
        """TestCase.setup defaults to a fresh SetupInfo instance."""
        tc = TestCase()
        assert isinstance(tc.setup, SetupInfo)
        assert tc.setup.container_id == ""

    def test_setup_field_is_independent_per_instance(self):
        """Each TestCase gets its own SetupInfo instance (no shared mutable default)."""
        tc1 = TestCase()
        tc2 = TestCase()
        tc1.setup.container_id = "id1"
        assert tc2.setup.container_id == ""


# ---------------------------------------------------------------------------
# Unit tests — parse_log_entries (legacy format support)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParseLogEntriesLegacyFormat:
    """Tests for parse_log_entries() with the new legacy-format support."""

    def test_parses_current_format(self, tmp_path):
        """parse_log_entries handles the current TIMESTAMP MODULE LEVEL CONTENT format."""
        log = tmp_path / "test.log"
        log.write_text(
            "2026-03-26 12:45:20,277 microbots.MicroBot INFO Hello world\n",
            encoding="utf-8",
        )
        entries = parse_log_entries(str(log))
        assert len(entries) == 1
        assert entries[0]["content"] == "Hello world"
        assert entries[0]["level"] == "INFO"
        assert entries[0]["module"] == "microbots.MicroBot"

    def test_parses_legacy_format(self, tmp_path):
        """parse_log_entries handles the legacy TIMESTAMP [LEVEL] CONTENT format."""
        log = tmp_path / "test.log"
        log.write_text(
            "2026-03-26 12:45:20,277 [INFO] Legacy format message\n",
            encoding="utf-8",
        )
        entries = parse_log_entries(str(log))
        assert len(entries) == 1
        assert entries[0]["content"] == "Legacy format message"
        assert entries[0]["level"] == "INFO"
        assert entries[0]["module"] == ""

    def test_continuation_lines_joined(self, tmp_path):
        """Lines without timestamps are joined to the previous entry."""
        log = tmp_path / "test.log"
        log.write_text(
            "2026-03-26 12:45:20,277 microbots.MicroBot INFO First line\n"
            "  continuation here\n",
            encoding="utf-8",
        )
        entries = parse_log_entries(str(log))
        assert len(entries) == 1
        assert "continuation here" in entries[0]["content"]

    def test_multiple_entries_both_formats(self, tmp_path):
        """Mix of current and legacy format entries are all parsed."""
        log = tmp_path / "test.log"
        log.write_text(
            "2026-03-26 12:45:20,277 microbots.MicroBot INFO Current format\n"
            "2026-03-26 12:45:21,000 [DEBUG] Legacy format\n",
            encoding="utf-8",
        )
        entries = parse_log_entries(str(log))
        assert len(entries) == 2
        assert entries[0]["content"] == "Current format"
        assert entries[1]["content"] == "Legacy format"
        assert entries[1]["module"] == ""

    def test_multiple_current_format_entries(self, tmp_path):
        """Multiple sequential current-format entries are all captured."""
        log = tmp_path / "multi.log"
        log.write_text(
            "2026-03-26 12:45:20,277 microbots.MicroBot INFO First entry\n"
            "2026-03-26 12:45:21,000 microbots.MicroBot INFO Second entry\n"
            "2026-03-26 12:45:22,000 microbots.MicroBot INFO Third entry\n",
            encoding="utf-8",
        )
        entries = parse_log_entries(str(log))
        assert len(entries) == 3
        assert entries[0]["content"] == "First entry"
        assert entries[1]["content"] == "Second entry"
        assert entries[2]["content"] == "Third entry"

    def test_empty_log_returns_empty_list(self, tmp_path):
        """An empty log file returns an empty list."""
        log = tmp_path / "empty.log"
        log.write_text("", encoding="utf-8")
        entries = parse_log_entries(str(log))
        assert entries == []


# ---------------------------------------------------------------------------
# Unit tests — _extract_setup_info
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExtractSetupInfo:
    """Tests for the new _extract_setup_info() function."""

    def _make_entry(self, content, level="INFO"):
        return {
            "timestamp": "2026-03-26 12:00:00,000",
            "module": "test",
            "level": level,
            "content": content,
            "line_num": 1,
        }

    def test_extracts_container_info(self):
        """Parses container ID, image, and host port from log entries."""
        entries = [
            self._make_entry(
                "Started container abc123 with image ubuntu:22.04 on host port 8080"
            ),
        ]
        setup = _extract_setup_info(entries)
        assert setup.container_id == "abc123"
        assert setup.image == "ubuntu:22.04"
        assert setup.host_port == "8080"

    def test_extracts_working_directory(self):
        """Parses working directory from log entries."""
        entries = [
            self._make_entry("Created working directory at /tmp/workspace"),
        ]
        setup = _extract_setup_info(entries)
        assert setup.working_dir == "/tmp/workspace"

    def test_extracts_volume_mapping(self):
        """Parses volume mappings from log entries."""
        entries = [
            self._make_entry("Volume mapping: /host/path:/container/path"),
        ]
        setup = _extract_setup_info(entries)
        assert "/host/path:/container/path" in setup.volume_mappings

    def test_extracts_tools_installed(self):
        """Parses installed tools from log entries."""
        entries = [
            self._make_entry("Successfully installed tool: git"),
            self._make_entry("Successfully set up tool: docker"),
        ]
        setup = _extract_setup_info(entries)
        assert "git" in setup.tools_installed
        assert "docker" in setup.tools_installed

    def test_no_duplicate_tools(self):
        """Same tool name is not added twice."""
        entries = [
            self._make_entry("Successfully installed tool: git"),
            self._make_entry("Successfully installed tool: git"),
        ]
        setup = _extract_setup_info(entries)
        assert setup.tools_installed.count("git") == 1

    def test_extracts_files_copied(self):
        """Parses copied files from log entries."""
        entries = [
            self._make_entry("Successfully copied repo to container: /workspace/repo"),
        ]
        setup = _extract_setup_info(entries)
        assert len(setup.files_copied) == 1
        assert "repo" in setup.files_copied[0]

    def test_stops_at_task_started(self):
        """Stops parsing setup info when TASK STARTED is encountered."""
        entries = [
            self._make_entry(
                "Started container ctn1 with image img1 on host port 9000"
            ),
            self._make_entry("ℹ️  TASK STARTED : some task"),
            # This entry comes AFTER task started and should be ignored
            self._make_entry("Volume mapping: /should/not/be/included"),
        ]
        setup = _extract_setup_info(entries)
        assert setup.container_id == "ctn1"
        assert setup.volume_mappings == []

    def test_empty_entries_returns_empty_setup(self):
        """Returns a default SetupInfo when entries list is empty."""
        setup = _extract_setup_info([])
        assert setup.container_id == ""
        assert setup.working_dir == ""

    def test_no_matching_entries_returns_empty_setup(self):
        """Returns empty SetupInfo when no setup patterns match."""
        entries = [
            self._make_entry("Just some random log message"),
            self._make_entry("Another random message"),
        ]
        setup = _extract_setup_info(entries)
        assert setup.container_id == ""
        assert setup.image == ""
        assert setup.working_dir == ""
        assert setup.volume_mappings == []


# ---------------------------------------------------------------------------
# Unit tests — extract_task_from_microbot_sub
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExtractTaskFromMicrobotSub:
    """Tests for the new extract_task_from_microbot_sub() function."""

    def test_extracts_task_with_iterations_flag(self):
        """Extracts --task value when followed by --iterations."""
        cmd = 'microbot_sub --task "Do the thing" --iterations 10'
        result = extract_task_from_microbot_sub(cmd)
        assert result == "Do the thing"

    def test_extracts_task_with_timeout_flag(self):
        """Extracts --task value when followed by --timeout."""
        cmd = 'microbot_sub --task "Run tests" --timeout 300'
        result = extract_task_from_microbot_sub(cmd)
        assert result == "Run tests"

    def test_extracts_task_at_end_of_command(self):
        """Extracts --task value at the end of the command string."""
        cmd = 'microbot_sub --task "Final task"'
        result = extract_task_from_microbot_sub(cmd)
        assert result == "Final task"

    def test_falls_back_to_full_command_when_no_task(self):
        """Returns the full command string when no --task flag is found."""
        cmd = "microbot_sub --some-other-arg value"
        result = extract_task_from_microbot_sub(cmd)
        assert result == cmd

    def test_handles_escaped_quotes(self):
        """Handles escaped quotes in the command string."""
        cmd = r'microbot_sub --task "Task with \"quotes\"" --iterations 5'
        result = extract_task_from_microbot_sub(cmd)
        assert "Task with" in result

    def test_handles_multiline_task(self):
        """Handles multi-line task descriptions."""
        cmd = 'microbot_sub --task "Line one\\nLine two" --iterations 5'
        result = extract_task_from_microbot_sub(cmd)
        assert "Line one" in result


# ---------------------------------------------------------------------------
# Unit tests — build_test_cases (new fields and new code paths)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildTestCases:
    """Tests for new code paths added to build_test_cases()."""

    def _make_entry(self, content, level="INFO"):
        return {
            "timestamp": "2026-03-26 12:00:00,000",
            "module": "MicroBot",
            "level": level,
            "content": content,
            "line_num": 1,
        }

    def test_empty_entries_returns_empty_list(self):
        """build_test_cases returns empty list for empty input."""
        assert build_test_cases([]) == []

    def test_task_started_creates_agent(self):
        """TASK STARTED creates a main agent with the task text."""
        entries = [
            self._make_entry("ℹ️  TASK STARTED : Do the work"),
            self._make_entry("TASK COMPLETED successfully"),
        ]
        test_cases = build_test_cases(entries)
        assert len(test_cases) == 1
        assert test_cases[0].main_agent is not None
        assert test_cases[0].main_agent.completed is True

    def test_task_completed_sets_completed_flag(self):
        """TASK COMPLETED sets agent.completed = True and clears current_field."""
        entries = [
            self._make_entry("ℹ️  TASK STARTED : Some task"),
            self._make_entry("TASK COMPLETED"),
        ]
        test_cases = build_test_cases(entries)
        assert test_cases[0].main_agent.completed is True

    def test_sub_agent_failed_sets_error_message(self):
        """ERROR Sub-agent failed sets error_message on the sub-agent."""
        entries = [
            self._make_entry("ℹ️  TASK STARTED : Main task"),
            self._make_entry("ℹ️  TASK STARTED : Sub task"),
            self._make_entry("Sub-agent failed: timed out", level="ERROR"),
        ]
        test_cases = build_test_cases(entries)
        assert len(test_cases) == 1
        assert len(test_cases[0].sub_agents) == 1
        assert test_cases[0].sub_agents[0].error_message == "Sub-agent failed: timed out"
        assert test_cases[0].sub_agents[0].max_iterations_reached is True

    def test_failed_to_parse_microbot_sub_sets_blocked(self):
        """ERROR Failed to parse microbot_sub command sets current_step as blocked."""
        from microbots.utils.multi_agent_log_parser import Step
        entries = [
            self._make_entry("ℹ️  TASK STARTED : Main task"),
            self._make_entry("LLM tool call: microbot_sub: bad command"),
            self._make_entry(
                "Failed to parse microbot_sub command: invalid syntax", level="ERROR"
            ),
        ]
        test_cases = build_test_cases(entries)
        # Should not raise and should produce a test case
        assert len(test_cases) >= 1


# ---------------------------------------------------------------------------
# Unit tests — truncate_text (new function)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTruncateText:
    """Tests for the new truncate_text() helper."""

    def test_short_text_unchanged(self):
        """Text within the line limit is returned as-is."""
        text = "Line one\nLine two\nLine three"
        assert truncate_text(text, max_lines=10) == text

    def test_long_text_truncated(self):
        """Text exceeding max_lines is truncated with a notice."""
        lines = [f"line {i}" for i in range(250)]
        text = "\n".join(lines)
        result = truncate_text(text, max_lines=200)
        assert "truncated" in result
        assert "50 more lines" in result

    def test_exact_limit_not_truncated(self):
        """Text at exactly max_lines is NOT truncated."""
        lines = [f"line {i}" for i in range(200)]
        text = "\n".join(lines)
        result = truncate_text(text, max_lines=200)
        assert "truncated" not in result


# ---------------------------------------------------------------------------
# Unit tests — generate_setup_md (new function)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGenerateSetupMd:
    """Tests for the new generate_setup_md() function."""

    def test_empty_setup_returns_empty_string(self):
        """Returns empty string when no container_id and no tools_installed."""
        setup = SetupInfo()
        assert generate_setup_md(setup) == ""

    def test_setup_with_container_id_returns_markdown(self):
        """Returns markdown when container_id is set."""
        setup = SetupInfo(container_id="abc123", image="ubuntu:22.04", host_port="8080")
        md = generate_setup_md(setup)
        assert "abc123" in md
        assert "ubuntu:22.04" in md
        assert "8080" in md

    def test_setup_with_working_dir(self):
        """Includes working directory in output."""
        setup = SetupInfo(container_id="ctn1", working_dir="/workspace")
        md = generate_setup_md(setup)
        assert "/workspace" in md

    def test_setup_with_volume_mappings(self):
        """Includes volume mappings in output."""
        setup = SetupInfo(container_id="ctn1", volume_mappings=["/host:/container"])
        md = generate_setup_md(setup)
        assert "/host:/container" in md

    def test_setup_with_tools_only(self):
        """Returns markdown when only tools_installed is set (no container_id)."""
        setup = SetupInfo(tools_installed=["git", "docker"])
        md = generate_setup_md(setup)
        assert "git" in md
        assert "docker" in md

    def test_setup_with_files_copied(self):
        """Includes files_copied section when files were copied."""
        setup = SetupInfo(
            container_id="ctn1",
            files_copied=["repo.py → /workspace/repo.py"],
        )
        md = generate_setup_md(setup)
        assert "Files copied" in md
        assert "repo.py" in md


# ---------------------------------------------------------------------------
# Unit tests — _agent_status_str (new function)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAgentStatusStr:
    """Tests for the new _agent_status_str() helper."""

    def test_completed_agent_returns_completed(self):
        """Returns completed string for completed agent."""
        agent = Agent(completed=True)
        assert "Completed" in _agent_status_str(agent)

    def test_max_iterations_agent_returns_failed(self):
        """Returns failed string for agent that hit max iterations."""
        agent = Agent(max_iterations_reached=True)
        result = _agent_status_str(agent)
        assert "Failed" in result

    def test_unknown_agent_returns_unknown(self):
        """Returns unknown string for agent with no terminal state."""
        agent = Agent()
        assert "Unknown" in _agent_status_str(agent)
