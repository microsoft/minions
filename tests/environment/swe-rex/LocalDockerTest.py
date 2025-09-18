"""Integration tests for LocalDocker validating real Docker container lifecycle, mounts, permissions, and command execution.

Run with: pytest -m docker test/environment/swe-rex/LocalDockerTest.py -q
Skip with: pytest -m 'not docker'
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterable
import sys

# Ensure project root is on sys.path for 'Environment' package resolution
PROJECT_ROOT = Path(__file__).parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

# Dynamic import due to hyphen in directory name
import importlib.util
LOCAL_DOCKER_PATH = Path(__file__).parents[3] / 'Environment' / 'swe-rex' / 'LocalDocker.py'
spec = importlib.util.spec_from_file_location('Environment.swe_rex.LocalDocker', LOCAL_DOCKER_PATH)
assert spec and spec.loader
_local_mod = importlib.util.module_from_spec(spec)  # type: ignore
spec.loader.exec_module(_local_mod)  # type: ignore
LocalDocker = getattr(_local_mod, 'LocalDocker')
Permission = getattr(_local_mod, 'Permission')

DOCKER_MARK = pytest.mark.docker

# Helpers

def docker_available() -> bool:
    return subprocess.call(['which', 'docker'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0

def list_running_container_ids() -> set[str]:
    out = subprocess.check_output(['docker', 'ps', '-q'], text=True).strip()
    return set(filter(None, out.splitlines()))


def find_new_container(before: set[str], after: set[str]) -> str | None:
    diff = after - before
    return next(iter(diff)) if diff else None


def inspect_container(container_id: str) -> dict:
    output = subprocess.check_output(['docker', 'inspect', container_id], text=True)
    return json.loads(output)[0]


@pytest.fixture(scope='module')
def ensure_docker():
    if not docker_available():
        pytest.skip('Docker CLI not available')


@pytest.fixture
def mounted_tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        # create some structure
        (p / 'subdir').mkdir()
        (p / 'subdir' / 'file.txt').write_text('hello world', encoding='utf-8')
        yield p


@pytest.fixture
def localdocker_rw(ensure_docker, mounted_tmp_dir):
    before = list_running_container_ids()
    env = LocalDocker(folder_to_mount=str(mounted_tmp_dir), permission=Permission.READ_WRITE)
    after = list_running_container_ids()
    cid = find_new_container(before, after)
    assert cid, 'No new container detected for RW env'
    yield env, cid, mounted_tmp_dir
    env.stop()
    # allow docker a moment to tear down
    time.sleep(0.5)
    assert cid not in list_running_container_ids(), 'Container still running after stop()'


@pytest.fixture
def localdocker_ro(ensure_docker, mounted_tmp_dir):
    before = list_running_container_ids()
    env = LocalDocker(folder_to_mount=str(mounted_tmp_dir), permission=Permission.READ_ONLY)
    after = list_running_container_ids()
    cid = find_new_container(before, after)
    assert cid, 'No new container detected for RO env'
    yield env, cid, mounted_tmp_dir
    env.stop()
    time.sleep(0.5)
    assert cid not in list_running_container_ids(), 'Container still running after stop()'


@DOCKER_MARK
def test_container_starts_and_stops(localdocker_rw):
    env, cid, mounted = localdocker_rw
    info = inspect_container(cid)
    assert info['State']['Running'] is True
    # Basic image check
    assert 'python:3.11' in info['Config']['Image']


@DOCKER_MARK
def test_mount_present_rw(localdocker_rw):
    env, cid, mounted = localdocker_rw
    info = inspect_container(cid)
    mount_points = info['Mounts']
    # locate our mount
    target_basename = mounted.name
    target_entry = next((m for m in mount_points if m['Source'] == str(mounted)), None)
    assert target_entry, f'Mount not found in container: {mount_points}'
    readonly_flag = target_entry.get('RW')  # docker inspect uses RW boolean (True means writable)
    assert readonly_flag is True


@DOCKER_MARK
def test_mount_present_ro(localdocker_ro):
    env, cid, mounted = localdocker_ro
    info = inspect_container(cid)
    target_entry = next((m for m in info['Mounts'] if m['Source'] == str(mounted)), None)
    assert target_entry, 'RO mount not found'
    assert target_entry.get('RW') is False


@DOCKER_MARK
def test_write_reflected_rw(localdocker_rw):
    env, cid, mounted = localdocker_rw
    # Path inside container
    inside_path = f"{env.BASE_PATH}/{mounted.name}/subdir/new_file.txt"
    result = asyncio.run(env.execute(f"bash -c 'echo from_container > {inside_path}'"))
    assert result.return_code == 0, result.stderr
    # Verify on host
    host_file = mounted / 'subdir' / 'new_file.txt'
    assert host_file.exists()
    assert host_file.read_text(encoding='utf-8').strip() == 'from_container'


@DOCKER_MARK
def test_write_blocked_ro(localdocker_ro):
    env, cid, mounted = localdocker_ro
    inside_path = f"{env.BASE_PATH}/{mounted.name}/subdir/should_fail.txt"
    result = asyncio.run(env.execute(f"bash -c 'echo test > {inside_path}'"))
    # Expect failure: non-zero return code or error string
    assert result.return_code != 0 or 'denied' in result.stderr.lower() or 'read-only' in result.stderr.lower()
    assert not (mounted / 'subdir' / 'should_fail.txt').exists()


@DOCKER_MARK
def test_directory_structure_visible(localdocker_rw):
    env, cid, mounted = localdocker_rw
    inside_dir = f"{env.BASE_PATH}/{mounted.name}/subdir"
    result = asyncio.run(env.execute(f"ls -1 {inside_dir}"))
    assert result.return_code == 0
    # Should list existing file.txt and possibly new_file.txt if prior test ran first (order independence not guaranteed)
    assert 'file.txt' in result.stdout


@DOCKER_MARK
def test_timeout_behavior(localdocker_rw):
    env, cid, mounted = localdocker_rw
    result = asyncio.run(env.execute("sleep 2", timeout=0.5))
    assert result.return_code == 124
    assert 'timed out' in result.stderr.lower()


@DOCKER_MARK
def test_multiple_commands_separate_calls(localdocker_rw):
    env, cid, mounted = localdocker_rw
    first = asyncio.run(env.execute("echo first"))
    second = asyncio.run(env.execute("echo second"))
    assert first.return_code == 0
    assert second.return_code == 0
    assert 'first' in first.stdout
    assert 'second' in second.stdout


@DOCKER_MARK
def test_invalid_permission_combo():
    with pytest.raises(ValueError):
        LocalDocker(folder_to_mount=None, permission=Permission.READ_ONLY)
    with pytest.raises(ValueError):
        LocalDocker(folder_to_mount=str(Path('/tmp')), permission=None)
