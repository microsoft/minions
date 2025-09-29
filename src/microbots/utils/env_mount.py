from dataclasses import dataclass, field
from enum import StrEnum

from microbots.constants import PermissionLabels, PermissionMapping
from microbots.utils.path import PathInfo, get_path_info


class MountType(StrEnum):
    MOUNT = "mount"
    COPY = "copy"


@dataclass
class Mount:
    host_path: str
    container_path: str
    permission: PermissionLabels
    mount_type: MountType = MountType.MOUNT

    # These will be set in __post_init__
    permission_key: str = field(init=False)
    host_path_info: PathInfo = field(init=False)

    def __post_init__(self):
        self.permission_key = PermissionMapping.MAPPING.get(self.permission)
        self.host_path_info = get_path_info(self.host_path)
        if not self.host_path_info.path_valid:
            raise ValueError(f"Path {self.host_path} is not a valid path")

    @property
    def container_full_path(self):
        return f"{self.container_path}/{self.host_path_info.base_name}"
