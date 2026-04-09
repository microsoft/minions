# Blog Series: AI-Powered Pipelines

A progressive article series that introduces Microbots to developers through practical pipeline use cases. The series is split into three tracks:

- **Core Series (Articles 1–5):** A linear learning path from first bot to production pipeline. After these 5 articles, a developer can solve real pipeline tasks with Microbots.
- **Advanced Articles (Articles 6–10):** Self-contained, ad-hoc deep dives into advanced features. Each article explains one advanced problem in detail. Can be read in any order.
- **Architecture Series:** Internals of Microbots with code references. Separate from the blog series, linked from Articles 1 and 2 for readers who want to understand the design in depth.

---

## Core Series (Articles 1–5)
    
*A linear path from zero to solving real pipeline tasks.*

---

### Article 1 — Getting Started with Microbots: AI Agents for Developer Pipelines

**Goal:** Get the reader from zero to a working bot in one article. Similar to the README but framed for pipeline use cases.

**Outline:**

- **What is Microbots?** A lightweight, extensible AI agent framework for code comprehension and controlled file edits. It integrates into automation pipelines, mounting a target directory with explicit read-only or read/write modes so LLMs can safely inspect, refactor, or generate files with least-privilege access.

- **Safety features** (one paragraph, not a deep dive): Microbots runs every command inside a disposable Docker container with OverlayFS isolation, OS-level permission enforcement (`READ_ONLY`/`READ_WRITE`), dangerous command detection, and iteration budgets. For a detailed architecture walkthrough, see the [Architecture Series](#architecture-series).

- **Prerequisites:** Python 3.11+, Docker, an LLM provider (Azure OpenAI / Anthropic / Ollama)

- **Installation:**

```bash
pip install microbots
```

- **Configure LLM** (`.env` file):

```env
OPEN_AI_END_POINT=your-endpoint-url
OPEN_AI_KEY=your-api-key
```

- **First bot — ReadingBot** (read-only, safe starting point):

```python
from microbots import ReadingBot

bot = ReadingBot(
    model="azure-openai/my-gpt5",
    folder_to_mount="my-project",
)

result = bot.run(
    "Analyze the test files and explain why test_auth.py is failing",
    timeout_in_seconds=300,
)
print(result.status)   # True/False
print(result.result)   # The bot's findings
print(result.error)    # None if successful
```

- **Explain `BotRunResult`**: `status`, `result`, `error`

- **WritingBot** (read/write, for applying fixes):

```python
from microbots import WritingBot

bot = WritingBot(
    model="azure-openai/my-gpt5",
    folder_to_mount="my-project",
)

result = bot.run(
    "Fix the syntax error in main.py and verify the fix runs correctly",
    timeout_in_seconds=600,
)
print(result.result)
```

- **Available bots** — brief table pointing to internals docs for details:

| Bot | Permission | Use Case |
|-----|-----------|----------|
| ReadingBot | READ_ONLY | Code analysis, debugging, review |
| WritingBot | READ_WRITE | Bug fixes, code generation, refactoring |
| BrowsingBot | — | Web research |
| LogAnalysisBot | READ_ONLY | Log root-cause analysis |
| AgentBoss | READ_WRITE | Multi-agent orchestration |
| MicroBot | Configurable | Custom workflows |

- **Supported LLM providers:** `azure-openai/<deployment>`, `anthropic/<deployment>`, `ollama-local/<model>`

- **Link to:** Architecture Series for safety deep dive, Article 2 for real pipeline integration

---

### Article 2 — Integrate LogAnalysisBot and WritingBot into Your CI/CD Pipeline

**Goal:** The natural follow-up to Article 1. Build a complete pipeline integration: LogAnalysisBot analyzes a failure → WritingBot applies the fix → pipeline re-triggers (with guardrails).

**Outline:**

- **The problem:** Pipeline fails → developer manually reads logs → context-switches to IDE → finds the bug → fixes → pushes → waits for pipeline again. This costs 15–45 minutes per failure.

- **The solution:** On pipeline failure, LogAnalysisBot reads the logs with source code context, diagnoses the root cause, then WritingBot applies the fix automatically.

- **LogAnalysisBot** — reads logs + code (both READ_ONLY):

```python
from microbots import LogAnalysisBot

bot = LogAnalysisBot(
    model="azure-openai/my-gpt5",
    folder_to_mount="src/",
)

result = bot.run(
    file_name="pipeline-output.log",
    max_iterations=20,
    timeout_in_seconds=300,
)
print(result.result)  # Root-cause analysis
```

- Explain how LogAnalysisBot works internally:
  - Source code mounted as READ_ONLY via OverlayFS
  - Log file COPY-mounted to `/var/log/` inside the container (isolated from host)
  - The bot cross-references log errors against source code to identify root cause

- **Chain to WritingBot** — propagate the analysis for an automated fix:

```python
from microbots import LogAnalysisBot, WritingBot

# Step 1: Diagnose
analyzer = LogAnalysisBot(
    model="azure-openai/my-gpt5",
    folder_to_mount="src/",
)
diagnosis = analyzer.run(file_name="pipeline-output.log")

# Step 2: Fix (only if diagnosis succeeded)
if diagnosis.status:
    fixer = WritingBot(
        model="azure-openai/my-gpt5",
        folder_to_mount="src/",
    )
    fix_result = fixer.run(
        f"Based on this root-cause analysis, apply the fix:\n{diagnosis.result}",
        max_iterations=25,
        timeout_in_seconds=600,
    )
    print(fix_result.result)
```

- **Pipeline integration** — Azure Pipeline YAML / GitHub Actions snippet that:
  1. Catches the failure
  2. Runs LogAnalysisBot
  3. Runs WritingBot
  4. Commits the fix to a branch
  5. Re-triggers the pipeline

- **Guardrails in the pipeline:**
  - WritingBot runs inside a container (never touches the host)
  - Iteration budget prevents runaway API costs
  - Dangerous command detection blocks destructive operations
  - The fix goes to a separate branch, not directly to main

- **Link to:** Internals docs for different bot types, Article 3 for custom bots

---

### Article 3 — Build a Custom Bot with MicroBot for Your Specific Workflow

**Goal:** Show that the user is not limited to pre-built bots. MicroBot is the base class — users can control the system prompt, mount configuration, and behavior entirely.

**Outline:**

- **Why customize?** Pre-built bots cover common cases, but every team has unique needs: custom linting rules, domain-specific analysis, proprietary build systems

- **MicroBot direct usage** — full control over system prompt and mounts:

```python
from microbots import MicroBot
from microbots.extras.mount import Mount
from microbots.constants import PermissionLabels

mount = Mount(
    host_path="/path/to/my-project",
    sandbox_path="/workdir/my-project",
    permission=PermissionLabels.READ_WRITE,
)

bot = MicroBot(
    model="azure-openai/my-gpt5",
    system_prompt="""You are a security auditor. Analyze the codebase for
    hardcoded credentials, insecure API calls, and missing input validation.
    Report findings with file paths and line numbers.""",
    folder_to_mount=mount,
)

result = bot.run(
    "Audit the authentication module for security vulnerabilities",
    max_iterations=30,
    timeout_in_seconds=600,
)
print(result.result)
```

- **Explain the Mount system:**
  - `host_path`: the directory on the host machine
  - `sandbox_path`: where it appears inside the container
  - `permission`: `READ_ONLY` or `READ_WRITE` (OS-enforced)
  - `mount_type`: `MOUNT` (bind mount) or `COPY` (file copy into container)

- **Multiple mount example** — mount code READ_WRITE + config READ_ONLY:

```python
from microbots.extras.mount import Mount, MountType

code_mount = Mount(
    "/path/to/src", "/workdir/src", PermissionLabels.READ_WRITE
)
config_mount = Mount(
    "/path/to/configs", "/workdir/configs", PermissionLabels.READ_ONLY
)
```

- **Custom system prompts** — explain how the system prompt shapes bot behavior:
  - Pre-built bots use carefully crafted prompts for their specific use case
  - CustomBot lets you write your own prompt for domain-specific reasoning
  - The system prompt includes tool usage instructions automatically when `additional_tools` are provided

- **Pipeline example:** A custom bot that validates Kubernetes manifests against company policies

- **Link to:** Article 4 for adding custom tools, internals docs for MicroBot class reference

---

### Article 4 — Extend Your Bot with Custom Tools for Specialized Debugging

**Goal:** Show how to attach specialized tools (gdb, tcpdump, Node.js crash debuggers, etc.) to any bot using the YAML tool definition system.

**Outline:**

- **Why tools?** Bots execute shell commands in the sandbox, but specialized debugging tools need installation, configuration, and usage instructions for the LLM.

- **The tool system:** Two types:
  - **Internal tools** — installed and run inside the Docker sandbox (linters, debuggers, analyzers)
  - **External tools** — run on the host machine (sub-agents, memory)

- **Defining a custom tool in YAML** — example: a Node.js crash analyzer:

```yaml
# nodejs-crash-analyzer.yaml
name: node-crash-analyzer
tool_type: internal
description: Analyzes Node.js crash dumps and core files

install_commands:
  - npm install -g node-report llnode
  - apt-get install -y gdb

verify_commands:
  - node --version
  - gdb --version

usage_instructions_to_llm: |
  # Node.js Crash Analysis Tool
  # Use these commands to analyze Node.js crashes:
  #
  # Generate a diagnostic report from a running process:
  #   node --report --report-filename=report.json <script.js>
  #
  # Analyze a crash dump with llnode:
  #   llnode -- node --abort-on-uncaught-exception <script.js>
  #
  # Use gdb for native crash analysis:
  #   gdb -batch -ex "bt full" -ex "info threads" node core.<pid>
  #
  # Always check the report JSON for:
  #   - javascriptStack: the JS call stack at crash time
  #   - nativeStack: the native C++ call stack
  #   - resourceUsage: memory and CPU at crash time
```

- **Attach to a bot:**

```python
from microbots import MicroBot
from microbots.tools.tool_yaml_parser import parse_tool_definition

crash_tool = parse_tool_definition("nodejs-crash-analyzer.yaml")

bot = MicroBot(
    model="azure-openai/my-gpt5",
    system_prompt="You are a Node.js crash debugging specialist...",
    folder_to_mount=mount,
    additional_tools=[crash_tool],
)

result = bot.run("Analyze the crash dump in /workdir/app/core.12345")
```

- **Tool lifecycle explained:**
  1. `install_commands` — run once when the bot starts (install packages)
  2. `verify_commands` — confirm installation succeeded
  3. `setup_commands` — run after code is mounted, before LLM starts
  4. `usage_instructions_to_llm` — appended to the system prompt so the LLM knows how to use the tool
  5. `uninstall_commands` — cleanup on teardown

- **Another example: tcpdump for network debugging:**

```yaml
name: packet-capture
tool_type: internal
description: Capture and analyze network traffic

install_commands:
  - apt-get install -y tcpdump tshark

verify_commands:
  - tcpdump --version

usage_instructions_to_llm: |
  # Capture packets (non-interactive, time-limited):
  #   tcpdump -i any -c 100 -w /tmp/capture.pcap
  # Read a capture file:
  #   tcpdump -r /tmp/capture.pcap -nn
  # Filter by port:
  #   tcpdump -r /tmp/capture.pcap port 443 -nn
```

- **`files_to_copy`** — copy custom scripts into the sandbox:

```yaml
files_to_copy:
  - src: analyze_crash.sh
    dest: /usr/local/bin/analyze_crash
    permissions: 7  # rwx for owner
```

- **Built-in tools** to mention: `cscope` (C/C++ code browsing), `browser-use` (web browsing)

- **Pipeline example:** CI detects a Node.js process crash → bot with crash analyzer tool diagnoses the segfault → reports root cause

- **Link to:** Article 5 for BrowsingBot, internals docs for tool system architecture

---

### Article 5 — Use BrowsingBot to Fetch External Context for Pipeline Decisions

**Goal:** Introduce BrowsingBot for web research. Use the CVE backporting example — the bot fetches CVE details from the web and cross-references against the codebase.

**Outline:**

- **Why external context?** Not all answers are in the codebase:
  - A dependency update breaks the build — the fix is in the changelog on GitHub
  - A CVE is published — the pipeline should check if the project is affected
  - A framework deprecates an API — the migration guide is on the official docs

- **BrowsingBot** — no folder mount, purpose-built for web research:

```python
from microbots import BrowsingBot

bot = BrowsingBot(model="azure-openai/my-gpt5")

result = bot.run(
    "Find the details of CVE-2026-1234. What versions are affected? "
    "What is the recommended fix or patch?",
    timeout_in_seconds=300,
)
print(result.result)
```

- **How it works:**
  - Uses `browser-use` (Playwright + LLM) running inside Docker
  - The browser never touches the host machine
  - BrowsingBot sends a single command to the container; the browser agent handles navigation autonomously

- **CVE backporting pipeline** — the complete use case:

```python
from microbots import BrowsingBot, ReadingBot, WritingBot

# Step 1: Fetch CVE details from the web
researcher = BrowsingBot(model="azure-openai/my-gpt5")
cve_info = researcher.run(
    "Find details of CVE-2026-1234: affected versions, severity, and recommended fix"
)

# Step 2: Check if the project is affected
analyzer = ReadingBot(
    model="azure-openai/my-gpt5",
    folder_to_mount="src/",
)
impact = analyzer.run(
    f"Based on this CVE information, check if our project is affected:\n"
    f"{cve_info.result}\n"
    f"Check dependency versions and affected code paths.",
)

# Step 3: Apply the patch if affected
if "affected" in impact.result.lower():
    fixer = WritingBot(
        model="azure-openai/my-gpt5",
        folder_to_mount="src/",
    )
    fixer.run(
        f"Apply the fix for CVE-2026-1234 based on:\n"
        f"CVE details: {cve_info.result}\n"
        f"Impact analysis: {impact.result}",
    )
```

- **Azure AD token injection** for BrowsingBot — when using Azure AD auth, the bot injects a fresh token into the container automatically

- **Pipeline example:** Nightly security scan → BrowsingBot fetches latest CVEs → ReadingBot checks impact → WritingBot patches → PR created

- **After Article 5:** The reader can solve real pipeline tasks: log analysis, code fixes, custom debugging, web research, and multi-bot chaining. The following advanced articles cover specialized topics.

- **Link to:** Advanced articles for memory, multi-agent, auth, and more

---

## Advanced Articles (Articles 6–10)

*Self-contained deep dives. Each article explains one advanced problem in detail. Can be read in any order.*

---

### Article 6 — Give Your Bot Long-Term Memory with the Memory Tool

**Goal:** Explain the Memory Tool — persistent state across agentic steps and pipeline runs.

**Outline:**

- **The problem:** Bots are stateless by default. Each `run()` invocation starts from scratch. Multi-step investigations lose intermediate results. Nightly pipelines cannot build on yesterday's findings.

- **The Memory Tool:** A file-backed persistence layer under `/memories/` on the host (`~/.microbots/memory/`).

- **Commands:**

| Command | Usage | Description |
|---------|-------|-------------|
| `view` | `memory view /memories/notes.md` | View a file or list a directory |
| `create` | `memory create /memories/progress.md "## Status\n- Found bug"` | Create or overwrite a file |
| `str_replace` | `memory str_replace /memories/progress.md --old "line 42" --new "line 45"` | Replace text in a file |
| `insert` | `memory insert /memories/progress.md --line 0 --text "# Header"` | Insert a line at a position |
| `delete` | `memory delete /memories/old_notes.md` | Delete a file or directory |
| `rename` | `memory rename /memories/draft.md /memories/final.md` | Move or rename a file |
| `clear` | `memory clear` | Clear all memory files |

- **Memory protocol:** Bots always check `/memories` first for earlier progress before starting work

- **Example: Multi-stage pipeline with memory:**

```python
from microbots import MicroBot
from microbots.tools.tool_definitions.memory_tool import MemoryTool

memory = MemoryTool()

# Stage 1: Analysis bot saves findings
analysis_bot = MicroBot(
    model="azure-openai/my-gpt5",
    system_prompt="Analyze the failing tests. Save your findings to /memories/analysis.md",
    folder_to_mount=mount,
    additional_tools=[memory],
)
analysis_bot.run("Find all failing tests and save root causes to memory")

# Stage 2: Fix bot reads findings and applies fixes
fix_bot = MicroBot(
    model="azure-openai/my-gpt5",
    system_prompt="Read /memories/analysis.md and fix the issues listed there.",
    folder_to_mount=mount,
    additional_tools=[memory],
)
fix_bot.run("Read the analysis from memory and apply fixes")
```

- **Use case: Nightly regression tracker** — bot saves test results to memory each night, compares against previous runs to detect regressions over time

---

### Article 7 — Orchestrate Multi-Agent Workflows with AgentBoss

**Goal:** Explain the multi-agent model — AgentBoss decomposes complex tasks and delegates to sub-agents.

**Outline:**

- **The problem:** Complex pipeline failures span multiple services, files, or systems. A single bot hits iteration limits or loses context.

- **AgentBoss** — a leadership bot that breaks tasks into subtasks and delegates:

```python
from microbots import AgentBoss

boss = AgentBoss(
    model="azure-openai/my-gpt5",
    folder_to_mount="monorepo/",
)

result = boss.run(
    "The CI pipeline is failing across three services. Analyze each service's "
    "test failures independently, identify root causes, and fix them.",
    max_iterations=50,
    timeout_in_seconds=900,
)
print(result.result)
```

- **How it works:**
  1. AgentBoss receives the complex task
  2. LLM decomposes it into numbered subtasks
  3. Each subtask is delegated via `microbot_sub --task "..." --iterations 25 --timeout 300`
  4. Each `microbot_sub` call spawns an autonomous MicroBot in the same sandbox
  5. AgentBoss reviews each result before proceeding
  6. Failed subtasks are retried with corrected instructions
  7. Final synthesis of all results

- **Iteration budget sharing:** If AgentBoss has 50 iterations and a sub-agent uses 30, only 20 remain for subsequent agents. This prevents runaway costs.

- **Pipeline example:** Monorepo CI failure → AgentBoss decomposes by service → sub-agents analyze each → AgentBoss synthesizes a unified report and applies fixes

---

### Article 8 — Extend the Dangerous Command Detection List

**Goal:** Show how to customize the safety layer by adding team-specific dangerous command patterns.

**Outline:**

- **The built-in patterns:** Microbots blocks `rm -rf`, `ls -R`, `tree`, `find` without `-maxdepth`, `rm --recursive` — with explanations and safer alternatives

- **Why extend?** Teams have their own dangerous patterns:
  - `DROP TABLE` or `DELETE FROM` without `WHERE` in database-connected pipelines
  - `kubectl delete namespace` in Kubernetes workflows
  - `git push --force` to protected branches
  - `docker rm -f` on production containers

- **How the detection works** — walk through the regex-based pattern matching in `MicroBot._get_dangerous_command_explanation()` and `_is_safe_command()`

- **How to extend** — add custom patterns by subclassing MicroBot or modifying the detection list

- **Each pattern includes:**
  - `pattern`: regex to match
  - `reason`: why it is dangerous
  - `alternative`: a safer command the LLM should use instead

- **The feedback loop:** Blocked commands are not silently dropped. The LLM receives the reason and alternative, learns to self-correct in subsequent iterations

---

### Article 9 — Use Azure Entra ID Authentication in Azure Pipelines

**Goal:** Production-ready Azure AD auth setup for pipeline environments — no static API keys.

**Outline:**

- **The problem:** Static API keys in pipeline variables are a security risk. Enterprise policies require managed identity or service principal auth.

- **Azure AD authentication methods:**

  - **Managed Identity** (recommended for Azure-hosted pipelines):

  ```bash
  export AZURE_AUTH_METHOD=azure_ad
  # No other env vars needed on Azure VMs, Container Apps, App Service
  ```

  - **Service Principal** (for non-Azure environments):

  ```bash
  export AZURE_AUTH_METHOD=azure_ad
  export AZURE_CLIENT_ID="your-client-id"
  export AZURE_TENANT_ID="your-tenant-id"
  export AZURE_CLIENT_SECRET="your-client-secret"
  ```

  - **Programmatic token provider:**

  ```python
  from azure.identity import DefaultAzureCredential, get_bearer_token_provider
  from microbots import WritingBot

  credential = DefaultAzureCredential()
  token_provider = get_bearer_token_provider(
      credential, "https://cognitiveservices.azure.com/.default"
  )

  bot = WritingBot(
      model="azure-openai/my-deployment",
      folder_to_mount="src/",
      token_provider=token_provider,
  )
  ```

- **Token refresh:** `get_bearer_token_provider` caches and proactively refreshes. Tasks are never interrupted by token expiration.

- **BrowsingBot with Azure AD:** The bot injects a fresh token into the Docker container as `AZURE_OPENAI_AD_TOKEN` — no API key needed inside the sandbox.

- **Azure Pipeline YAML example:** Complete pipeline definition using managed identity with Microbots

- **Install the optional dependency:**

```bash
pip install microbots[azure_ad]
```

---

### Article 10 — Build a CoPilot Bot with Bring Your Own Key (BYOK)

**Goal:** Create a CoPilot-style bot where users bring their own LLM API keys and interact with Microbots through a custom interface.

**Outline:**

- **The concept:** A hosted service where each user provides their own LLM API key (BYOK). The service wraps Microbots and exposes a simplified interface — users get AI-powered code assistance without the team managing LLM costs.

- **Architecture:**
  - User provides their API key via a config or UI
  - The service creates a bot with the user's key
  - All execution happens in isolated containers per user
  - Keys are never stored — passed as `token_provider` or environment variable per session

- **Implementation pattern:**

```python
from microbots import MicroBot
from microbots.extras.mount import Mount
from microbots.constants import PermissionLabels

def create_copilot_session(user_api_key: str, project_path: str):
    """Create an isolated bot session for a user with their own API key."""
    import os
    os.environ["OPEN_AI_KEY"] = user_api_key

    mount = Mount(project_path, "/workdir/project", PermissionLabels.READ_WRITE)

    bot = MicroBot(
        model="azure-openai/user-deployment",
        system_prompt="You are a helpful coding assistant...",
        folder_to_mount=mount,
    )
    return bot
```

- **Multi-provider support:** Users can choose `azure-openai`, `anthropic`, or `ollama-local` — the BYOK model supports all providers

- **Security considerations:**
  - Each user's bot runs in its own Docker container — full isolation
  - User keys are scoped to the session and never persisted
  - READ_ONLY mount by default, READ_WRITE only when explicitly requested

- **Tool attachment:** Allow users to select which tools to enable (code analysis, web search, memory)

---

## Architecture Series

*Separate from the blog series. Internals of Microbots with code references. Linked from Articles 1 and 2.*

---

### Architecture Article 1 — Complete Architecture and Safety Features

**Goal:** Deep dive into the full architecture with diagrams and code references.

**Outline:**

- High-level architecture diagram: User → Bot → LLM + Docker Container
- Core components: `MicroBot`, `LLMInterface`, `Environment`, `ToolAbstract`, `Mount`
- Data flow: user task → system prompt + task → LLM → JSON response (`task_done`, `thoughts`, `command`) → sandbox execution → output → LLM → loop
- The 5 safety layers in detail with code references:
  1. Container isolation — `LocalDockerEnvironment`, shell server on port 8080
  2. OverlayFS — lower/upper layer setup, `_setup_overlay_mount()`, `_teardown_overlay_mount()`
  3. Permission labels — `PermissionLabels`, `Mount`, OS-level enforcement
  4. Dangerous command detection — `_get_dangerous_command_explanation()`, regex patterns, feedback to LLM
  5. Iteration budget — `iteration_count`, `max_iterations`, parent-child propagation

---

### Architecture Article 2 — Abstraction Layers: LLM, Environment, Tools, and Cost Management

**Goal:** Explain each abstraction layer with internal code references.

**Outline:**

- **LLM Layer:**
  - `LLMInterface` ABC: `ask()`, `clear_history()`, `_validate_llm_response()`
  - `OpenAIApi` — Azure OpenAI with `responses.create`
  - `AnthropicApi` — Anthropic / Azure AI Foundry with `messages.create`
  - `OllamaLocal` — local HTTP API to Ollama server
  - Response validation: JSON format enforcement, retry logic, structured `LLMAskResponse`

- **Environment Layer:**
  - `Environment` ABC: `start()`, `stop()`, `execute()`, `copy_to_container()`
  - `LocalDockerEnvironment` — Docker container with shell server, OverlayFS setup
  - `CmdReturn` dataclass: `stdout`, `stderr`, `return_code`
  - Working directory management: `~/MICROBOTS_WORKDIR_<random>`

- **Tool Layer:**
  - `ToolAbstract` (ABC) → `Tool` (internal, runs in sandbox) vs `ExternalTool` (runs on host)
  - YAML tool definitions via `parse_tool_definition()`
  - Tool lifecycle: install → verify → setup → (LLM uses) → uninstall
  - `usage_instructions_to_llm` — appended to system prompt
  - `files_to_copy` with `EnvFileCopies` — copy scripts with specific permissions

- **Cost Management:**
  - Iteration budgets: `max_iterations` per bot, shared across parent-child
  - Timeout enforcement: `timeout_in_seconds`
  - Token-level cost: varies by provider and model — choose cheaper models for simpler tasks

- **Dangerous Command Detection:**
  - Regex-based pattern list in `_get_dangerous_command_explanation()`
  - Patterns: `rm -rf`, `ls -R`, `tree`, `find` without `-maxdepth`, `rm --recursive`
  - Each pattern has `reason` and `alternative`
  - Feedback loop: LLM receives explanation and self-corrects

---

## Series Summary

| #  | Track | Title | Microbots Features |
|----|-------|-------|--------------------|
| 1  | Core | Getting Started with Microbots | ReadingBot, WritingBot, BotRunResult, installation, LLM config |
| 2  | Core | LogAnalysisBot + WritingBot in CI/CD | LogAnalysisBot, WritingBot, Mount (COPY), pipeline integration |
| 3  | Core | Build a Custom Bot with MicroBot | MicroBot base class, custom system_prompt, Mount, PermissionLabels |
| 4  | Core | Custom Tools for Specialized Debugging | YAML tool definitions, install/verify/setup lifecycle, files_to_copy |
| 5  | Core | BrowsingBot for External Context | BrowsingBot, browser-use, multi-bot chaining, CVE pipeline |
| 6  | Advanced | Long-Term Memory with Memory Tool | MemoryTool, cross-stage persistence |
| 7  | Advanced | Multi-Agent Workflows with AgentBoss | AgentBoss, microbot_sub, iteration budget sharing |
| 8  | Advanced | Extend Dangerous Command Detection | Safety patterns, regex matching, custom rules |
| 9  | Advanced | Azure Entra ID Auth in Pipelines | Azure AD, token_provider, managed identity, `microbots[azure_ad]` |
| 10 | Advanced | CoPilot Bot with BYOK | MicroBot, multi-provider, session isolation |
| A1 | Architecture | Complete Architecture & Safety | All internals, 5 safety layers, code references |
| A2 | Architecture | Abstraction Layers | LLM, Environment, Tools, Cost Management layers |

---

## Writing & Publishing Plan

| Track | Articles | Approach | Cadence |
|-------|----------|----------|---------|
| Core (1–5) | Linear series | One writer, sequential | Publish as a batch or weekly |
| Advanced (6–10) | Independent articles | One writer per article, in parallel | Ad-hoc, one at a time |
| Architecture (A1–A2) | Reference docs | Can start in parallel with Core | Publish before or alongside Article 1 |

**Cross-linking rules:**
- Article 1 links to Architecture Series for safety deep dive
- Article 2 links to internals docs for bot types
- Articles 3–5 link to previous articles as prerequisites
- Advanced articles are self-contained but link to Core articles for context
- All articles link to the relevant docs pages from the documentation site
