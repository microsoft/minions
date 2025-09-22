import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FolderMountInfo:
    path_valid: bool
    base_name: str
    abs_path: str


def get_file_mount_info(file_to_mount: str) -> FolderMountInfo:
    return_value = FolderMountInfo(
        path_valid=False,
        base_name="",
        abs_path="",
    )
    if is_absolute_path(file_to_mount):
        return_value.path_valid = is_valid_path(file_to_mount)
        return_value.abs_path = file_to_mount
        return_value.base_name = os.path.basename(file_to_mount)
    else:
        return_value.abs_path = get_absolute_path(file_to_mount)
        return_value.path_valid = is_valid_path(return_value.abs_path)
        return_value.base_name = file_to_mount

    return return_value


def get_folder_mount_info(folder_to_mount: str) -> FolderMountInfo:
    return_value = FolderMountInfo(
        path_valid=False,
        base_name="",
        abs_path="",
    )
    if is_absolute_path(folder_to_mount):
        return_value.path_valid = is_valid_path(folder_to_mount)
        return_value.abs_path = folder_to_mount
        return_value.base_name = get_base_name(folder_to_mount)
    else:
        return_value.abs_path = get_absolute_path(folder_to_mount)
        return_value.path_valid = is_valid_path(return_value.abs_path)
        return_value.base_name = folder_to_mount

    return return_value


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
