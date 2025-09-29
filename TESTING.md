# Integration Testing Guide for Microbots

This guide explains how to structure and run integration tests that simulate how users would import and use the microbots library.

## Overview

The integration tests are designed to:
1. **Validate the public API** - Ensure all exported classes and methods work as expected
2. **Test user workflows** - Simulate real user scenarios from import to execution
3. **Verify library structure** - Confirm that imports work correctly for end users
4. **Test different usage patterns** - Cover various ways users might use the library

## Directory Structure

```
tests/
├── integration/           # User-facing integration tests
│   ├── base_test.py      # Base test utilities and setup
│   ├── test_microbots_api.py              # Public API validation
│   ├── test_reading_bot_integration.py    # ReadingBot user scenarios
│   ├── test_writing_bot_integration.py    # WritingBot user scenarios
│   ├── test_browsing_bot_integration.py   # BrowsingBot user scenarios
│   └── test_log_analysis_bot_integration.py # LogAnalysisBot user scenarios
│
validate_installation.py   # Quick validation script
demo_user_experience.py    # Demonstration script
run_integration_tests.py   # Interactive test runner
```

## Test Categories

### 1. API Validation Tests (`test_microbots_api.py`)
- **Purpose**: Validate library structure without making API calls
- **What they test**:
  - Import patterns (`from microbots import ReadingBot`)
  - Class interfaces (methods exist and are callable)
  - Parameter validation
  - Error handling
- **Run without**: API keys, Docker, internet connection

### 2. Bot Integration Tests (`test_*_bot_integration.py`)
- **Purpose**: Test complete user workflows
- **What they test**:
  - Real bot initialization and execution
  - File system operations
  - API integrations
  - Result processing
- **Requires**: API keys, Docker, internet (for BrowsingBot)

## Running Tests

### Option 1: Quick Validation (No API Keys Required)
```bash
# Validate that the library is properly installed and importable
python validate_installation.py

# Demo the user experience without API calls
python demo_user_experience.py
```

### Option 2: Interactive Test Runner
```bash
# Run the interactive test selection menu
python run_integration_tests.py
```

Choose from:
1. **API Validation Tests** - Import and interface tests (no API calls)
2. **Individual Bot Tests** - Test specific bots with API calls
3. **All Integration Tests** - Complete test suite

### Option 3: Direct Test Execution
```bash
# Run API validation only (fast, no API keys needed)
python -m pytest tests/integration/test_microbots_api.py -v

# Run specific bot tests (requires API keys)  
python -m pytest tests/integration/test_reading_bot_integration.py -v

# Run all integration tests
python -m pytest tests/integration/ -v
```

## Environment Setup

### Required for API Tests
Create a `.env` file in the project root:
```env
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

### Docker Requirement
- Install Docker Desktop
- Ensure Docker daemon is running
- The tests will automatically pull required containers

## Test Structure Philosophy

### User-Centric Design
Tests are written from the user's perspective:

```python
# This is how users import the library
from microbots import ReadingBot

# This is how users would initialize
reading_bot = ReadingBot(
    model="azure-openai/gpt-4",
    folder_to_mount="/path/to/project"
)

# This is how users would use it
result = reading_bot.run("What does this code do?")
print(result.result)
```

### Real Scenarios
Tests use realistic scenarios:
- **ReadingBot**: Analyze actual code projects
- **WritingBot**: Fix bugs, add features, create new files
- **BrowsingBot**: Search for current information
- **LogAnalysisBot**: Analyze real log files with patterns

### Comprehensive Coverage
Each bot test covers:
- Basic usage patterns
- Advanced scenarios
- Error conditions
- Different parameter combinations

## Writing New Integration Tests

### Base Test Class
Extend `MicrobotsIntegrationTestBase` for utilities:

```python
from tests.integration.base_test import MicrobotsIntegrationTestBase, requires_azure_openai

class TestMyFeature(MicrobotsIntegrationTestBase):
    
    @requires_azure_openai
    def test_my_scenario(self):
        # Create test workspace
        workspace = self.create_test_workspace("my_test")
        
        # Create sample files
        self.create_sample_files(workspace, {
            "test.py": "print('hello')"
        })
        
        # Test your scenario
        # ... test code here ...
```

### Test Decorators
- `@requires_azure_openai` - Skip if API keys not available
- Mark slow tests appropriately
- Use descriptive test names

### Test Data
- Create realistic sample files
- Use temporary directories (auto-cleaned)
- Test both success and failure scenarios

## Continuous Integration

### Test Matrix
```yaml
# Example CI configuration
matrix:
  include:
    - name: "API Validation"
      run: python validate_installation.py
      requires: none
    
    - name: "Integration Tests"  
      run: python -m pytest tests/integration/
      requires: api_keys, docker
```

### Performance Considerations
- API validation tests: ~30 seconds
- Individual bot tests: 2-5 minutes each
- Full integration suite: 10-15 minutes

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the project root
   cd /path/to/microbots/project
   python validate_installation.py
   ```

2. **API Key Issues**
   ```bash
   # Check environment variables
   python -c "import os; print('API Key set:', bool(os.getenv('AZURE_OPENAI_API_KEY')))"
   ```

3. **Docker Issues**
   ```bash
   # Verify Docker is running
   docker ps
   
   # Check Docker access
   docker run hello-world
   ```

4. **Test Failures**
   ```bash
   # Run with verbose output
   python -m pytest tests/integration/test_microbots_api.py -v -s
   
   # Run specific test
   python -m pytest tests/integration/test_microbots_api.py::TestMicrobotsPublicAPI::test_main_imports -v
   ```

## Best Practices

### For Test Writers
1. **Test user workflows**, not internal implementation
2. **Use realistic scenarios** that users would actually encounter
3. **Validate both success and error paths**
4. **Clean up resources** (files, containers) after tests
5. **Make tests independent** - each test should be runnable alone

### For Users
1. **Start with validation** - `python validate_installation.py`
2. **Run API tests first** - Quick feedback without API calls
3. **Test incrementally** - One bot at a time
4. **Check logs** - Use `-v` flag for detailed output
5. **Verify environment** - Ensure all prerequisites are met

This testing approach ensures that the microbots library works exactly as users expect, catching integration issues that unit tests might miss.