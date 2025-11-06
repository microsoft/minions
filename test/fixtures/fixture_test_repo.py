import pytest
from pathlib import Path
import subprocess
import os

TEST_REPO = "https://github.com/SWE-agent/test-repo"

@pytest.fixture
def test_repo(tmpdir):
    # Check is root exists
    assert tmpdir.exists()

    try:
        result = subprocess.run(
            ["git", "-C", str(tmpdir), "clone", "--depth", "1", TEST_REPO],
            check=True,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            pytest.fail(
                f"Failed to clone repository: {TEST_REPO}\n"
                f"Return code: {result.returncode}\n"
                f"Stdout: {result.stdout}\n"
                f"Stderr: {result.stderr}"
            )

    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"Exception while cloning repository: {TEST_REPO}\n"
            f"Exception: {str(e)}\n"
        )

    repo_path = Path(tmpdir / os.listdir(tmpdir)[0])
    yield repo_path

    # Cleanup after test
    if repo_path.exists():
        subprocess.run(["rm", "-rf", str(repo_path)])
