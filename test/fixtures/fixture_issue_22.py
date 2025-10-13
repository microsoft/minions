import pytest

ISSUE_TEXT = """
How to reproduce:

```
from testpkg.tribonacci import tribonacci


assert tribonacci(0) == 0
```
"""

@pytest.fixture
def issue_22():
    return ISSUE_TEXT, None