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
        subprocess.run(["git", "-C", str(tmpdir), "clone", TEST_REPO], check=True)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Failed to clone repository: {e}")

    repo_path = Path(os.path.abspath(os.listdir(tmpdir)[0]))
    yield repo_path

    # Cleanup after test
    if repo_path.exists():
        subprocess.run(["rm", "-rf", str(repo_path)])
