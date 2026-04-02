"""Utility to read GitHub Copilot CLI credentials from ~/.copilot/config.json."""

import json
from logging import getLogger
from pathlib import Path
from typing import Optional

logger = getLogger(__name__)

COPILOT_CONFIG_PATH = Path.home() / ".copilot" / "config.json"


def get_copilot_token(config_path: Path = COPILOT_CONFIG_PATH) -> Optional[str]:
    """Extract the OAuth token from the Copilot CLI config file.

    The Copilot CLI stores credentials in ``~/.copilot/config.json`` after
    ``copilot auth login``.  This function reads the first available token
    from the ``copilot_tokens`` map.

    Returns ``None`` if the file doesn't exist or contains no tokens.
    """
    if not config_path.is_file():
        logger.debug("Copilot config not found at %s", config_path)
        return None

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read Copilot config at %s: %s", config_path, exc)
        return None

    tokens = data.get("copilot_tokens", {})
    if not tokens:
        logger.debug("No copilot_tokens found in %s", config_path)
        return None

    # Return the first available token
    token = next(iter(tokens.values()))
    logger.debug("Resolved Copilot token from %s", config_path)
    return token
