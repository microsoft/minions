"""Token provider utilities for dynamic API key / bearer token management."""

import os
import time
from collections.abc import Callable
from logging import getLogger

logger = getLogger(__name__)

TokenProvider = Callable[[], str]
"""A callable that returns a valid API key or bearer token."""


def env_token_provider(var_name: str) -> TokenProvider:
    """Return a TokenProvider that reads from an environment variable each time."""
    def _provider() -> str:
        value = os.getenv(var_name)
        if not value:
            raise ValueError(f"Environment variable '{var_name}' is not set or empty")
        return value
    return _provider


def static_token_provider(token: str) -> TokenProvider:
    """Return a TokenProvider that always returns the same fixed token."""
    def _provider() -> str:
        return token
    return _provider


class CachedTokenProvider:
    """Wraps a TokenProvider with a time-based cache.

    The underlying provider is only called when the cached token has expired.

    Parameters
    ----------
    provider : TokenProvider
        The underlying callable that fetches a fresh token.
    ttl_seconds : float
        How long (in seconds) a cached token is considered valid.
        Defaults to 300 (5 minutes). Set to 0 to disable caching.
    """

    def __init__(self, provider: TokenProvider, ttl_seconds: float = 300):
        self._provider = provider
        self._ttl = ttl_seconds
        self._cached_token: str | None = None
        self._fetched_at: float = 0

    def __call__(self) -> str:
        now = time.monotonic()
        if self._cached_token is None or (now - self._fetched_at) >= self._ttl:
            logger.debug("Token cache expired or empty, fetching fresh token")
            self._cached_token = self._provider()
            self._fetched_at = now
        return self._cached_token
