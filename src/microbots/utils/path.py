import os
from pathlib import Path


def is_valid_path(path: str) -> bool:
    try:
        return Path(path).exists()
    except Exception:
        return False


def is_absolute_path(path: str) -> bool:
    return Path(path).is_absolute()


def get_base_name(path: str) -> str:
    return Path(path).name


def get_absolute_path(path: str) -> str:
    return str(Path(path).resolve(strict=False))
