"""Unit tests for microbots.utils.copilot_auth.get_copilot_token."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

from microbots.utils.copilot_auth import get_copilot_token


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetCopilotToken:
    """Tests for get_copilot_token()."""

    def test_returns_none_when_file_missing(self, tmp_path):
        """Returns None when the config file does not exist."""
        missing = tmp_path / "nonexistent.json"
        assert get_copilot_token(config_path=missing) is None

    def test_returns_none_on_invalid_json(self, tmp_path):
        """Returns None and logs a warning when the file contains invalid JSON."""
        bad_file = tmp_path / "config.json"
        bad_file.write_text("this is not json", encoding="utf-8")
        assert get_copilot_token(config_path=bad_file) is None

    def test_returns_none_when_no_copilot_tokens_key(self, tmp_path):
        """Returns None when the JSON has no 'copilot_tokens' key."""
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"other_key": "value"}), encoding="utf-8")
        assert get_copilot_token(config_path=cfg) is None

    def test_returns_none_when_copilot_tokens_empty(self, tmp_path):
        """Returns None when 'copilot_tokens' is an empty dict."""
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"copilot_tokens": {}}), encoding="utf-8")
        assert get_copilot_token(config_path=cfg) is None

    def test_returns_first_token(self, tmp_path):
        """Returns the first token value from 'copilot_tokens'."""
        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps({"copilot_tokens": {"host1": "token-abc", "host2": "token-xyz"}}),
            encoding="utf-8",
        )
        token = get_copilot_token(config_path=cfg)
        assert token == "token-abc"

    def test_returns_none_on_os_error(self, tmp_path):
        """Returns None when the file cannot be read (OSError)."""
        cfg = tmp_path / "config.json"
        cfg.write_text("{}", encoding="utf-8")
        cfg.chmod(0o000)  # remove read permission
        try:
            result = get_copilot_token(config_path=cfg)
            assert result is None
        finally:
            cfg.chmod(0o644)  # restore permissions for cleanup
