import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PathInfo:
    path_valid: bool
    base_name: str
    abs_path: str


def is_valid_path(path: str) -> bool:
    try:
        return Path(path).exists()
    except Exception:
        raise ValueError(f"Path {path} does not exist!")


def is_absolute_path(path: str) -> bool:
    return Path(path).is_absolute()


def get_base_name(path: str) -> str:
    return Path(path).name


def get_absolute_path(path: str) -> str:
    return str(Path(path).resolve(strict=False))


def get_path_info(file_or_folder: str) -> PathInfo:
    if is_valid_path(file_or_folder):
        return PathInfo(
            path_valid=True,
            base_name=os.path.basename(file_or_folder),
            abs_path=os.path.abspath(file_or_folder),
        )
    return PathInfo(path_valid=False, base_name="", abs_path="")


def ends_with_separator(path_str: str) -> bool:
    # Accept both separators on Windows; only '/' on POSIX
    seps = {os.sep}
    if os.altsep:
        seps.add(os.altsep)
    return len(path_str) > 0 and path_str[-1] in seps

