# Running Tests Individually

This document explains how to run tests individually or retrigger failed tests while maintaining proper code coverage reporting.

## Overview

The project supports running tests in two modes:

1. **Standard CI Tests** - Runs all tests in the normal CI pipeline
2. **Individual/Failed Test Reruns** - Allows retriggering specific tests or only failed tests

## Running All Tests Locally

To run all tests locally with coverage:

```bash
# Run all unit tests
pytest -m 'unit' --cov=src --cov-report=xml --cov-report=term-missing

# Run all integration tests
pytest -m 'integration' --cov=src --cov-report=xml --cov-report=term-missing

# Run all tests
pytest --cov=src --cov-report=xml --cov-report=term-missing
```

## Running Individual Tests Locally

### Run a Specific Test File

```bash
pytest test/bot/test_microbot.py --cov=src --cov-report=xml
```

### Run a Specific Test Class

```bash
pytest test/bot/test_microbot.py::TestMicroBot --cov=src --cov-report=xml
```

### Run a Specific Test Method

```bash
pytest test/bot/test_microbot.py::TestMicroBot::test_microbot_ro_mount --cov=src --cov-report=xml
```

## Rerunning Only Failed Tests

Pytest automatically tracks which tests failed in the last run. To rerun only those tests:

```bash
# Rerun only last failed tests
pytest --lf --cov=src --cov-report=xml

# Rerun failed tests first, then all others
pytest --ff --cov=src --cov-report=xml
```

The test failure information is stored in `.pytest_cache/v/cache/lastfailed`.

## GitHub Actions: Rerun Failed Tests Workflow

A dedicated workflow is available for retriggering failed tests in CI:

### How to Use

1. Go to the **Actions** tab in the GitHub repository
2. Select the **"Rerun Failed Tests"** workflow from the left sidebar
3. Click **"Run workflow"** button
4. Configure the workflow inputs:
   - **test-type**: Choose which test type to run (`unit`, `integration`, or `all`)
   - **test-path**: (Optional) Specify a specific test file or test path
   - **last-failed-only**: (Default: true) Run only tests that failed in the last run

### Examples

#### Rerun All Last Failed Unit Tests
- test-type: `unit`
- test-path: (leave empty)
- last-failed-only: `true`

#### Rerun a Specific Failed Test
- test-type: `unit` or `integration`
- test-path: `test/bot/test_microbot.py::TestMicroBot::test_microbot_ro_mount`
- last-failed-only: `false`

#### Rerun All Tests in a File
- test-type: `unit` or `integration`
- test-path: `test/bot/test_microbot.py`
- last-failed-only: `false`

## Coverage Reports

### How Coverage Works

- Each test run generates a `coverage.xml` file
- The `--cov-append` flag allows multiple test runs to accumulate coverage
- Coverage reports are uploaded to Codecov with appropriate flags
- Different test types (unit/integration) have separate coverage flags for tracking

### Coverage with Individual Tests

When running individual tests:

1. **Single Test Run**: Coverage will only reflect the code paths exercised by that specific test
2. **Multiple Test Runs**: Use `--cov-append` to accumulate coverage:

```bash
# First test run
pytest test/bot/test_microbot.py::TestMicroBot::test_microbot_ro_mount --cov=src --cov-report=xml

# Additional test runs - appends to existing coverage
pytest test/bot/test_reading_bot.py --cov=src --cov-report=xml --cov-append
```

3. **Combined Coverage**: To see total coverage after multiple runs:

```bash
# Generate combined report
coverage combine
coverage report
coverage xml
```

### Codecov Integration

- The main test workflow uploads coverage with flags: `unit` or `integration`
- The rerun workflow uploads coverage with flags: `unit,rerun` or `integration,rerun`
- Codecov automatically combines coverage from multiple uploads
- You can view coverage by flag in the Codecov dashboard

## Best Practices

1. **Use pytest cache for efficiency**: The `.pytest_cache` directory is cached in CI to speed up reruns
2. **Run failed tests first**: Use `--lf` to quickly validate fixes without running the entire suite
3. **Maintain coverage**: When fixing failed tests, ensure new code paths are covered
4. **Review coverage reports**: Check Codecov to ensure individual test runs don't decrease overall coverage

## Troubleshooting

### "No tests collected" when using --lf

This means no tests failed in the previous run. Either:
- All tests passed previously
- The `.pytest_cache` directory was cleared
- This is the first test run

Solution: Run without `--lf` flag or use `--lfnf=all` to run all tests if no failures are found.

### Coverage seems incomplete

When running individual tests, coverage only reflects those specific tests. To get full coverage:
- Run the complete test suite, or
- Use `--cov-append` across multiple test runs to accumulate coverage

### Pytest cache not found in CI

The pytest cache is uploaded as an artifact and cached between runs. If not found:
- Ensure the previous workflow run completed
- Check that the cache hasn't expired (GitHub caches expire after 7 days of no access)
- Run the main test workflow first to generate the cache
