# Microbots : Introduction, Installation Guide and Creating Your First MicroBot

**Published on:** April 15, 2026 | **Author:** Siva Kannan

!!! warning "External Links Disclaimer"
    This document contains links to external documentation that may change or be removed over time. All external references were verified at the time of writing and are only considered valid as of the publication date.

## Introduction

Microbots is a lightweight, extensible AI agent framework for code comprehension and controlled file edits. It integrates cleanly into automation pipelines, mounting a target directory with explicit read-only or read/write permissions so LLMs can safely inspect, refactor, or generate files with least-privilege access. Every command an agent executes runs inside a disposable Docker container — your host machine, files, and credentials are never exposed.

## Safety Features

Microbots enforces safety through five reinforcing layers: **container isolation** that runs every command in a disposable Docker container, **OverlayFS** for copy-on-write filesystem protection, **OS-level permission labels** (`READ_ONLY` / `READ_WRITE`) on every mounted folder, a **dangerous command detection** validator that blocks destructive patterns before execution, and **iteration budget management** that prevents runaway costs from sub-agents. The core philosophy is simple — assume the LLM will eventually produce a harmful command, and architect the system so that it does not matter when it does.

!!! tip "Want to understand the details?"
    Read the [Microbots : Safety First Agentic Workflow](../../blog/microbots-safety-first-ai-agent.md) article for a deep dive into the architecture and all five layers of defense.

## Pre-requisites

### Docker

Microbots runs all agent commands inside Docker containers, so Docker must be installed and running on your machine.

**Install Docker Desktop:**

- **Windows / macOS:** Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
- **Linux:** Follow the official [Docker Engine installation guide](https://docs.docker.com/engine/install/) for your distribution.

After installation, verify Docker is running:

```bash title="Terminal"
docker --version
docker run hello-world
```

If both commands succeed, Docker is ready.

### AI LLM Provider

Microbots currently supports **OpenAI**, **Anthropic**, and **Ollama** as LLM providers. In this guide, we will use **OpenAI** as the provider.

You will need an **API Key** (`OPEN_AI_KEY`), an **Endpoint URL** (`OPEN_AI_END_POINT`), and a **deployed model name**.

Create a `.env` file in the root of your application with:

```env title=".env"
OPEN_AI_END_POINT=https://your-resource-name.openai.azure.com
OPEN_AI_KEY=your-api-key-here
```

!!! note
    For advanced authentication options (Azure AD tokens, managed identity, service principals), see the [Authentication Guide](../advanced/authentication.md).

## Install Microbots

```bash title="Terminal"
pip install microbots
```

## Creating Your First MicroBot

### Step 1: Prepare a sample project

Create a project folder with a TypeScript file that has a syntax error. When the TypeScript compiler (`tsc`) tries to convert it to JavaScript, it fails — and the error output is captured in a log file.

```bash title="Terminal"
mkdir microbots-introduction
cd microbots-introduction

# Create a .env file with your Azure OpenAI credentials
# (see Pre-requisites above)

mkdir code
```

Create `code/app.ts` with a deliberate syntax error:

```typescript title="code/app.ts"
--8<-- "docs/examples/microbots_introduction/code/app.ts"
```

The function `add` on line 8 has a malformed type annotation (`b: number: number` instead of `b: number): number`). Run the TypeScript compiler to generate the build log:

```bash title="Terminal"
cd code
tsc app.ts > build.log 2>&1
cd ..
```

This produces the following build log:

```log title="code/build.log"
--8<-- "docs/examples/microbots_introduction/code/build.log"
```

Your folder structure should look like:

```
microbots-introduction/
├── .env
├── log_analysis_bot.py
└── code/
    ├── app.ts
    └── build.log
```

### Step 2: Analyze logs with a LogAnalysisBot

The `LogAnalysisBot` mounts the target folder as **read-only** and analyzes log files to identify the root cause of failures.

```python title="log_analysis_bot.py"
--8<-- "docs/examples/microbots_introduction/log_analysis_bot.py"
```

Run it:

```bash title="Terminal"
python log_analysis_bot.py
```

The `LogAnalysisBot` will spin up a Docker container, mount the `code` folder as read-only, and use the LLM to analyze the log file and report the root cause.

The output of `print(result.result)` will look something like:

```text title="Output"
The build failed due to a syntax error in app.ts on line 8. The function
add has a malformed type annotation — b: number: number should be
b: number): number. The missing closing parenthesis causes the TypeScript
compiler to produce cascading errors on lines 8–10. To fix this, change
line 8 to: function add(a: number, b: number): number {
```

The `LogAnalysisBot` read the `build.log`, correlated the compiler errors with the source code in `app.ts`, identified the malformed type annotation as the root cause, and provided a clear fix — all without any human intervention.

!!! tip "DevOps Integration"
    This pattern integrates naturally into CI/CD pipelines. Point the `LogAnalysisBot` at build logs, test reports, or deployment logs from tools like GitHub Actions, Azure DevOps, Jenkins, or GitLab CI — and get instant root-cause analysis delivered as part of your pipeline output.

### What just happened?

Behind the scenes, Microbots:

1. **Created a Docker container** with the `code` folder mounted using the appropriate permissions.
2. **Sent your task** to the LLM along with a system prompt tailored to the bot type.
3. **Executed commands** inside the container as directed by the LLM (e.g., `cat`, `grep`, `sed`).
4. **Returned the result** — the bot's analysis and root-cause report.

Your host filesystem was protected the entire time. The LogAnalysisBot physically could not write to your files — all within Docker's isolation boundary.

## Available Bots

Beyond the `LogAnalysisBot` used in this guide, Microbots provides several other bots tailored for different use cases — each with its own permission level to ensure least-privilege access.

| Bot | Permission | Description |
|-----|-----------|-------------|
| **ReadingBot** | Read-only | Reads files and extracts information based on instructions |
| **WritingBot** | Read-write | Reads and writes files to fix issues or generate code |
| **BrowsingBot** | — | Browses the web to gather information |
| **LogAnalysisBot** | Read-only | Analyzes logs for instant root-cause debugging |
| **AgentBoss** | — | Orchestrates multiple bots for complex multi-step tasks |