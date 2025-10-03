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

def verify_function(test_repo):
    # Run missing_colon.py and the return value should be 0
    try:
        result = subprocess.run(["python3", str(test_repo / "tests" / "missing_colon.py")], capture_output=True, text=True)
        assert result.returncode == 0
    except Exception as e:
        pytest.fail(f"Failed to verify function: {e}")


@pytest.fixture
def issue_1():
    return ISSUE_TEXT, verify_function
