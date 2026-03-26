"""Tests for the token_provider module."""

import os
import time
from unittest.mock import MagicMock

import pytest

from microbots.llm.token_provider import (
    CachedTokenProvider,
    env_token_provider,
    static_token_provider,
)


class TestStaticTokenProvider:
    def test_returns_fixed_token(self):
        provider = static_token_provider("my-token-123")
        assert provider() == "my-token-123"
        assert provider() == "my-token-123"


class TestEnvTokenProvider:
    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("TEST_TOKEN_VAR", "env-token-abc")
        provider = env_token_provider("TEST_TOKEN_VAR")
        assert provider() == "env-token-abc"

    def test_raises_when_not_set(self, monkeypatch):
        monkeypatch.delenv("TEST_TOKEN_VAR", raising=False)
        provider = env_token_provider("TEST_TOKEN_VAR")
        with pytest.raises(ValueError, match="not set or empty"):
            provider()

    def test_picks_up_updated_env(self, monkeypatch):
        monkeypatch.setenv("TEST_TOKEN_VAR", "old-token")
        provider = env_token_provider("TEST_TOKEN_VAR")
        assert provider() == "old-token"
        monkeypatch.setenv("TEST_TOKEN_VAR", "new-token")
        assert provider() == "new-token"


class TestCachedTokenProvider:
    def test_caches_within_ttl(self):
        mock = MagicMock(side_effect=["token-1", "token-2"])
        cached = CachedTokenProvider(mock, ttl_seconds=60)
        assert cached() == "token-1"
        assert cached() == "token-1"
        assert mock.call_count == 1

    def test_refreshes_after_ttl(self):
        mock = MagicMock(side_effect=["token-1", "token-2"])
        cached = CachedTokenProvider(mock, ttl_seconds=0.1)
        assert cached() == "token-1"
        time.sleep(0.15)
        assert cached() == "token-2"
        assert mock.call_count == 2

    def test_zero_ttl_always_refreshes(self):
        mock = MagicMock(side_effect=["t1", "t2", "t3"])
        cached = CachedTokenProvider(mock, ttl_seconds=0)
        assert cached() == "t1"
        assert cached() == "t2"
        assert cached() == "t3"
        assert mock.call_count == 3

    def test_callable_protocol(self):
        """CachedTokenProvider is callable and can be used as a TokenProvider."""
        cached = CachedTokenProvider(lambda: "hello")
        assert callable(cached)
        assert cached() == "hello"
