import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, f"{Path(__file__).resolve().parents[2].as_posix()}/src"
)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from microbots.extras.mount import Mount

@pytest.mark.integration
def test_non_abs_sandbox_path(tmpdir: Path):
    with pytest.raises(ValueError, match="sandbox_path must be an absolute path"):
        Mount(
            host_path=str(tmpdir),
            sandbox_path="relative/path/in/sandbox",
            permission="READ_ONLY",
        )

@pytest.mark.integration
def test_host_file_sandbox_dir(tmpdir: Path):
    host_file = tmpdir / "testfile.txt"
    host_file.write_text("Sample content", encoding="utf-8")

    mount = Mount(
        host_path=str(host_file),
        sandbox_path="/sandbox/dir/",
        permission="READ_ONLY",
        mount_type="copy",
    )

    assert mount.sandbox_path == "/sandbox/dir/testfile.txt"
