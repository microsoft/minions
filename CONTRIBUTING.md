# Contributing to MicroBots

We welcome contributions to the MicroBots project! This document provides guidelines for contributing to this repository.

## Ways to Contribute

There are many ways to contribute to MicroBots:

- **Report bugs** and help us verify fixes
- **Submit feature requests** for new bot types or capabilities
- **Improve documentation** to help others use MicroBots
- **Contribute code** with bug fixes or new features
- **Share examples** of how you're using MicroBots

## Reporting Issues

### Before Submitting an Issue

1. **Search existing issues** to avoid duplicates
2. **Check the documentation** in README.md and CLAUDE.md
3. **Verify the issue** with the latest version

### Creating a Good Bug Report

A good bug report should include:

- **Clear description** of the issue
- **Steps to reproduce** the problem
- **Expected behavior** vs actual behavior
- **Environment details**:
  - Python version
  - Docker version
  - Operating system
  - MicroBots version
- **Error messages** and relevant logs
- **Minimal code example** that demonstrates the issue

### Feature Requests

We welcome feature suggestions! When proposing a new feature:

- Explain the **use case** and why it's valuable
- Describe the **expected behavior**
- Consider how it fits with the project's goals
- Be open to discussion about alternative approaches

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Docker (required for LocalDockerEnvironment)
- Git

### Setting Up Your Development Environment

1. **Fork and clone** the repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/minions.git
   cd minions
   ```

2. **Create a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in development mode
   ```

4. **Configure environment variables**:
   Create a `.env` file with your Azure OpenAI credentials:

   ```env
   OPEN_AI_END_POINT=<your-endpoint>
   OPEN_AI_KEY=<your-api-key>
   ```

5. **Verify Docker is running**:

   ```bash
   docker ps
   ```

## Making Changes

### Creating a Branch

Create a feature branch for your work:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

Use descriptive branch names:

- `feature/browsing-bot-improvements`
- `fix/mount-permission-error`
- `docs/improve-readme`

### Code Style

- Follow [PEP 8](https://pep8.org/) Python style guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Include type hints where appropriate
- Use the existing logging patterns:

  ```python
  import logging
  logger = logging.getLogger(__name__)
  ```

### Testing Your Changes

1. **Run existing tests**:

   ```bash
   # Run all tests
   pytest

   # Run tests excluding Docker (faster)
   pytest -m "not docker"

   # Run specific test file
   pytest test/bot/calculator/log_analysis_test.py
   ```

2. **Test manually** with Docker:

   - Verify your changes work with actual Docker containers
   - Test both READ_ONLY and READ_WRITE permissions
   - Check error handling and edge cases

3. **Add new tests** for new functionality:
   - Place tests in the appropriate `test/` subdirectory
   - Follow existing test patterns
   - Mark Docker-dependent tests with `@pytest.mark.docker`

### Documentation

- Update README.md if you add user-facing features
- Update CLAUDE.md if you change architecture significantly
- Add docstrings to new functions and classes
- Include inline comments for complex logic

## Submitting a Pull Request

### Before Submitting

- [ ] Code follows the project style guidelines
- [ ] All tests pass locally
- [ ] New code has appropriate tests
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive

### Pull Request Process

1. **Push your branch** to your fork:

   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a pull request** on GitHub:

   - Provide a clear title describing the change
   - Reference any related issues (e.g., "Fixes #123")
   - Describe what changed and why
   - Include testing steps

3. **Pull request template**:

   ```markdown
   ## Description

   Brief description of the changes

   ## Type of Change

   - [ ] Bug fix (non-breaking change which fixes an issue)
   - [ ] New feature (non-breaking change which adds functionality)
   - [ ] Breaking change (fix or feature that would cause existing functionality to change)
   - [ ] Documentation update

   ## Related Issues

   Fixes #(issue number)

   ## Testing

   Describe the testing you performed

   ## Checklist

   - [ ] Tests pass locally
   - [ ] Code follows project style
   - [ ] Documentation updated
   ```

4. **Respond to feedback**:
   - Be open to suggestions and discussion
   - Make requested changes in new commits
   - Update your PR description if scope changes

### Commit Messages

Write clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "Fix security issue with shell=True in subprocess calls"
git commit -m "Add support for multiple directory mounts in bots"
git commit -m "Update OpenAI API to use chat.completions.create"

# Less helpful
git commit -m "fix bug"
git commit -m "updates"
```

## Project Structure

```bash
minions/
├── src/microbots/          # Main package source
│   ├── bot/                # Bot implementations
│   ├── environment/        # Environment abstractions
│   ├── llm/                # LLM API integrations
│   ├── tools/              # Tool system
│   └── utils/              # Utility functions
├── test/                   # Test files
├── docs/                   # Documentation
└── requirements.txt        # Dependencies
```

## Key Architectural Concepts

Before contributing, familiarize yourself with:

- **Bot types**: ReadingBot, WritingBot, LogAnalysisBot, etc.
- **Environment abstraction**: How Docker containers are managed
- **Mount system**: READ_ONLY vs READ_WRITE permissions
- **Tool installation**: YAML-based tool definitions
- **LLM interaction loop**: JSON response format requirements

See CLAUDE.md for detailed architecture documentation.

## Code Review Process

All submissions require review before merging:

1. Maintainers will review your code
2. Automated checks must pass
3. At least one approval required
4. Address any requested changes
5. Maintainer will merge when ready

## Questions?

- Open a [discussion](https://github.com/microsoft/minions/discussions) for general questions
- Use [issues](https://github.com/microsoft/minions/issues) for bug reports and feature requests
- Check existing documentation in README.md and CLAUDE.md

## License

By contributing to MicroBots, you agree that your contributions will be licensed under the MIT License.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Thank You

Your contributions help make MicroBots better for everyone. We appreciate your time and effort! 🤖
