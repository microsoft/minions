import pytest
import subprocess

ISSUE_TEXT = """
I'm running `tests/missing_colon.py` as follows:

division(23, 0)
but I get the following error:

```
  File "/Users/fuchur/Documents/24/git_sync/swe-agent-test-repo/tests/./missing_colon.py", line 4
    def division(a: float, b: float) -> float
                                             ^
SyntaxError: invalid syntax
```
"""

run_command = ["python3", "tests/missing_colon.py"]

def run_function(test_repo):
    result = subprocess.run(run_command, cwd=test_repo, capture_output=True, text=True)
    return result

def verify_function(test_repo):
    # Run missing_colon.py and the return value should be 0
    try:
        result = subprocess.run(run_command, cwd=test_repo, capture_output=True, text=True)
        assert result.returncode == 0
    except Exception as e:
        pytest.fail(f"Failed to verify function: {e}")


@pytest.fixture
def issue_1():
    return ISSUE_TEXT, verify_function, run_function
