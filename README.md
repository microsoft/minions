# 🤖 MicroBots

MicroBots is a lightweight, extensible AI agent framework for code comprehension and controlled file edits. It creates containerized environments that mount target directories with explicit read-only or read-write permissions, enabling LLMs to safely inspect, refactor, or generate files with least-privilege access.

## Quick Start

```py
from microbots import WritingBot

bot = WritingBot(
    model="azure-openai/my-gpt5",  # Format: <provider>/<deployment_name>
    folder_to_mount="myReactApp",
)

result = bot.run(
    "When doing npm run build, I get an error. Fix the error and make sure the build is successful.",
    timeout_in_seconds=600
)
print(result.result)
```

## ⚠️ Project Status: Highly Unstable

This project is currently **under active development** and is considered **highly unstable**. Features, APIs, and internal structures are subject to change without notice, and unexpected behavior may occur.
Please **use with caution** in production environments.

## Table of Contents

- [🤖 MicroBots](#-microbots)
  - [Quick Start](#quick-start)
  - [⚠️ Project Status: Highly Unstable](#️-project-status-highly-unstable)
  - [Table of Contents](#table-of-contents)
  - [🚀 Installation](#-installation)
    - [Prerequisites](#prerequisites)
    - [Install MicroBots](#install-microbots)
  - [✨ LLM Support](#-llm-support)
    - [Azure OpenAI](#azure-openai)
  - [🤖 Available Bots](#-available-bots)
    - [📖 ReadingBot](#-readingbot)
    - [✍️ WritingBot](#️-writingbot)
    - [🧪 LogAnalysisBot](#-loganalysisbot)
    - [🌐 BrowsingBot](#-browsingbot)
    - [🛠️ CustomBot](#️-custombot)
  - [🛠️ Tool System](#️-tool-system)
    - [Adding Tools to Bots](#adding-tools-to-bots)
    - [Tool Definition Format](#tool-definition-format)
    - [Built-in Tools](#built-in-tools)
  - [⚙️ How It Works](#️-how-it-works)
  - [🤝 Contributing](#-contributing)
    - [How to Contribute](#how-to-contribute)
  - [📄 License](#-license)
  - [🔗 Resources](#-resources)
  - [📝 Code of Conduct](#-code-of-conduct)

## 🚀 Installation

### Prerequisites

- **Docker** - Required for containerized execution environments
- **Python 3.11+** - Required for running MicroBots
- **AI LLM Provider and API Key** - Currently supports Azure OpenAI

### Install MicroBots

```bash
pip install microbots
```

## ✨ LLM Support

### Azure OpenAI

Create a `.env` file in your application root with your Azure OpenAI credentials:

```env
OPEN_AI_END_POINT=<your-azure-openai-endpoint>
OPEN_AI_KEY=<your-api-key>
```

## 🤖 Available Bots

MicroBots provides specialized bot types for different use cases. Each bot operates in an isolated Docker container with specific permissions and capabilities.

> **💡 Try the examples below:**
>
> To follow along with the examples, clone this test repository which contains sample Python files with intentional bugs:
>
> ```bash
> git clone https://github.com/swe-agent/test-repo/ code
> ```
>
> This creates a `code` folder with files like `missing_colon.py` (a Python file with a syntax error) that you can use to test how the bots find and fix issues.

### 📖 ReadingBot

Read-only access to analyze and understand code without making modifications.

```py
from microbots import ReadingBot

bot = ReadingBot(
    model="azure-openai/my-gpt5",
    folder_to_mount="code"
)

result = bot.run(
    "When I am running missing_colon.py I am getting SyntaxError: invalid syntax. Find the error and explain me what is the error",
    timeout_in_seconds=600
)
print(result.result)
```

**Use cases:** Code analysis, documentation generation, architecture review, bug investigation

### ✍️ WritingBot

Read-write access to modify, refactor, or generate code files.

```py
from microbots import WritingBot

bot = WritingBot(
    model="azure-openai/my-gpt5",
    folder_to_mount="code"
)

result = bot.run(
    "When I am running missing_colon.py I am getting SyntaxError: invalid syntax. Fix the error and make sure the code runs without any errors.",
    timeout_in_seconds=600
)
print(result.result)
```

**Use cases:** Bug fixes, code refactoring, feature implementation, test generation

### 🧪 LogAnalysisBot

Specialized bot for analyzing log files with read-only access to source code for correlation.

```py
from microbots import LogAnalysisBot

bot = LogAnalysisBot(
    model="azure-openai/my-gpt5",
    folder_to_mount="src"  # Source code directory (read-only)
)

result = bot.run(
    file_name="logs/application.log",  # Log file to analyze
    timeout_in_seconds=600
)
print(result.result)
```

**Use cases:** Root cause analysis, error investigation, performance debugging, failure diagnosis

### 🌐 BrowsingBot

Web browsing capabilities for gathering information from the internet.

```py
from microbots import BrowsingBot

bot = BrowsingBot(
    model="azure-openai/my-gpt5"
)

result = bot.run(
    "I need to understand async/await patterns in Python. Can you search for the latest best practices and examples?",
    timeout_in_seconds=600
)
print(result.result)
```

**Use cases:** API documentation lookup, framework research, library comparisons, technology trends

### 🛠️ CustomBot

Fully customizable bot with user-defined system prompts and optional directory mounting.

```py
from microbots import CustomBot

bot = CustomBot(
    model="azure-openai/my-gpt5",
    system_prompt="You are a security analyst. Review code for vulnerabilities and explain them clearly.",
    folder_to_mount="src"  # Optional, defaults to read-write if provided
)

result = bot.run(
    "I'm worried about security in the authentication module. Can you check for any vulnerabilities?",
    timeout_in_seconds=600
)
print(result.result)
```

**Use cases:** Custom workflows, specialized analysis, domain-specific tasks, experimental bots

## 🛠️ Tool System

MicroBots supports a YAML-based tool system for extending bot capabilities with additional tools and utilities.

### Adding Tools to Bots

```py
from microbots import WritingBot
from microbots.tools.tool import parse_tool_definition

# Load a custom tool definition
my_tool = parse_tool_definition("path/to/my-tool.yaml")

bot = WritingBot(
    model="azure-openai/my-gpt5",
    folder_to_mount="code",
    additional_tools=[my_tool]
)
```

### Tool Definition Format

Tools are defined in YAML with installation, verification, and usage instructions:

```yaml
name: my-tool
description: Description of what the tool does
installation:
  - command: apt-get update && apt-get install -y my-tool
verification:
  - command: which my-tool
setup:
  - command: my-tool --configure
environment_variables:
  MY_TOOL_CONFIG: /path/to/config
```

### Built-in Tools

- **cscope** - Code navigation and symbol search
- **browser-use** - Web browsing capabilities (used by BrowsingBot)

See `src/microbots/tools/` for tool examples.

## ⚙️ How It Works

![Overall Architecture Image](./docs/images/overall_architecture.png)

MicroBots creates isolated Docker containers with controlled access to your code:

1. **Environment Creation**: Spins up a Docker container with a Python execution environment
2. **Directory Mounting**: Mounts your specified directory with explicit permissions:
   - `READ_ONLY`: Uses overlay filesystem to prevent any modifications
   - `READ_WRITE`: Allows full file system access within the mounted directory
3. **Tool Installation**: Installs any additional tools specified in the bot configuration
4. **LLM Interaction Loop**:
   - Sends task to LLM with system prompt and available tools
   - Receives structured JSON response with commands to execute
   - Executes commands in the isolated container
   - Returns output to LLM for next iteration
5. **Result Extraction**: Returns final result when task is complete or timeout/max iterations reached
6. **Cleanup**: Stops and removes the Docker container

This architecture ensures AI agents operate within defined boundaries, enhancing security and control while protecting your local environment.

## 🤝 Contributing

We welcome contributions to MicroBots! Whether you're fixing bugs, adding features, improving documentation, or sharing examples, your help makes this project better.

### How to Contribute

- **Report Issues**: Found a bug? [Open an issue](https://github.com/microsoft/minions/issues)
- **Suggest Features**: Have an idea? [Start a discussion](https://github.com/microsoft/minions/discussions)
- **Submit PRs**: Ready to code? Check out our [Contributing Guide](CONTRIBUTING.md)
- **Share Examples**: Show us how you're using MicroBots!

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:

- Development setup
- Code style and testing
- Pull request process
- Project architecture

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Resources

- **Documentation**: See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation
- **Repository**: [github.com/microsoft/minions](https://github.com/microsoft/minions)
- **Issues**: [Bug Reports & Feature Requests](https://github.com/microsoft/minions/issues)
- **Discussions**: [Community Forum](https://github.com/microsoft/minions/discussions)

## 📝 Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

---

Built with ❤️ by the Microsoft team and contributors
