import logging
import pytest
from pathlib import Path


pytest_plugins = [
    "fixtures.fixture_test_repo",
    "fixtures.fixture_issue_1",
    "fixtures.fixture_issue_22",
    "llm.conftest",  # Make Ollama fixtures available to all tests
]


LOG_DIR = Path("/tmp/microbots_pytest_logs/")
# delete existing logs to start fresh for each test run
if LOG_DIR.exists():
    for log_file in LOG_DIR.iterdir():
        log_file.unlink()
LOG_DIR.mkdir(parents=True, exist_ok=True)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    # 1. Get the name of the test file (e.g., 'test_login' from 'test_login.py')
    test_filename = Path(item.fspath).stem

    # 2. Define your log paths
    info_log_name = f"{test_filename}_info.log"
    debug_log_name = f"{test_filename}_debug.log"

    # 3. Create a formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # 4. Set up INFO file handler
    info_handler = logging.FileHandler(LOG_DIR / info_log_name, mode='a')
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)

    # 5. Set up DEBUG file handler
    debug_handler = logging.FileHandler(LOG_DIR / debug_log_name, mode='a')
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)

    # 6. Attach handlers to the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Ensure root captures everything

    # Clear existing handlers to prevent duplicate entries from previous tests
    logger.handlers = [info_handler, debug_handler]