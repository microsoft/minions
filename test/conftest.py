pytest_plugins = [
    "fixtures.fixture_test_repo",
    "fixtures.fixture_issue_1",
    "fixtures.fixture_issue_22",
    "llm.conftest",  # Make Ollama fixtures available to all tests
]