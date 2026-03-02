#!/usr/bin/env python3
"""
Parse microbots info.log files into markdown trajectory files.

Usage:
    python multi_agent_log_parser.py <test_case>_info.log [output_dir]

Creates:
    <test_case>_trajectory/
        main_agent.md
        sub_agent_1.md
        sub_agent_2.md
        ...

The info.log file should be named as <test_case>_info.log.
A directory <test_case>_trajectory will be created with all the markdown files.
"""

import re
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────── Data Classes ───────────────────────────


@dataclass
class Step:
    """Represents a single step in an agent's execution."""
    number: int
    thought: str = ""
    command: str = ""
    output: str = ""
    is_blocked: bool = False
    blocked_reason: str = ""
    blocked_alternative: str = ""
    is_sub_agent_call: bool = False
    sub_agent_task: str = ""
    sub_agent_index: int = -1  # index into the test case's sub_agents list
    is_final: bool = False  # True if this represents LLM final thoughts


@dataclass
class Agent:
    """Represents an agent (main or sub) and its execution steps."""
    task: str = ""
    steps: List[Step] = field(default_factory=list)
    is_main: bool = False
    final_thoughts: str = ""
    completed: bool = False
    max_iterations_reached: bool = False


@dataclass
class TestCase:
    """Represents a single test case with a main agent and sub-agents."""
    name: str = ""
    main_agent: Optional[Agent] = None
    sub_agents: List[Agent] = field(default_factory=list)


# ─────────────────────────── Log Parsing ───────────────────────────

# Regex for parsing log line timestamps
LOG_LINE_RE = re.compile(
    r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[(INFO|ERROR|WARNING|DEBUG)\] (.*)$'
)


def parse_log_entries(log_path: str) -> List[dict]:
    """
    Parse a log file into a list of entries.
    Multi-line log entries (continuation lines without timestamps) are joined.

    Returns a list of dicts: {'timestamp': str, 'level': str, 'content': str, 'line_num': int}
    """
    entries = []
    current_entry = None

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.rstrip('\n')
            match = LOG_LINE_RE.match(line)
            if match:
                # Save previous entry
                if current_entry is not None:
                    entries.append(current_entry)
                current_entry = {
                    'timestamp': match.group(1),
                    'level': match.group(2),
                    'content': match.group(3),
                    'line_num': line_num,
                }
            else:
                # Continuation of previous entry
                if current_entry is not None:
                    current_entry['content'] += '\n' + line
                # else: lines before any log entry (skip)

    # Don't forget the last entry
    if current_entry is not None:
        entries.append(current_entry)

    return entries


# ─────────────────────────── Structure Building ───────────────────────────


def extract_task_from_microbot_sub(command: str) -> str:
    """Extract the --task argument from a microbot_sub command."""
    # Normalize escaped quotes: \" -> "
    normalized = command.replace('\\"', '"').replace('\\n', '\n')

    # Try to find --task "..." followed by " --iterations or end
    match = re.search(r'--task\s+"(.*?)"\s+--(?:iterations|timeout)', normalized, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find --task "..." at end of command
    match = re.search(r'--task\s+"(.*?)"\s*$', normalized, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try single quotes
    match = re.search(r"--task\s+'(.*?)'\s+--(?:iterations|timeout)", normalized, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: grab everything after --task " until the last " before --iterations
    match = re.search(r'--task\s+"(.+)', normalized, re.DOTALL)
    if match:
        text = match.group(1)
        # Try to find closing quote before --iterations or --timeout
        iter_match = re.search(r'"\s+--(?:iterations|timeout)', text)
        if iter_match:
            return text[:iter_match.start()].strip()
        # Try the last quote
        quote_end = text.rfind('"')
        if quote_end > 0:
            return text[:quote_end].strip()
        return text.strip()
    return command


def build_test_cases(entries: List[dict]) -> List[TestCase]:
    """
    Walk through log entries and build a list of TestCase objects,
    each containing a main agent and its sub-agents.
    """
    test_cases = []
    current_test: Optional[TestCase] = None

    # Agent tracking
    agent_stack: List[Agent] = []  # stack: [main_agent, sub_agent, ...]
    current_step: Optional[Step] = None
    pending_sub_agent_step: Optional[Step] = None  # main agent step that called microbot_sub
    current_field: Optional[str] = None  # track what we're collecting multi-line for

    def current_agent() -> Optional[Agent]:
        return agent_stack[-1] if agent_stack else None

    def finalize_test_case():
        nonlocal current_test, agent_stack, current_step, pending_sub_agent_step, current_field
        if current_test and current_test.main_agent:
            test_cases.append(current_test)
        current_test = None
        agent_stack = []
        current_step = None
        pending_sub_agent_step = None
        current_field = None

    for entry in entries:
        content = entry['content']
        level = entry['level']

        # ── Skip noise ──
        if 'HTTP Request:' in content:
            continue
        if content.startswith('The llm response is'):
            continue

        # ── Test case boundary ──
        test_dir_match = re.search(r'Test directory set up at:\s*\S+/(\S+)', content)
        if test_dir_match:
            finalize_test_case()
            test_name = test_dir_match.group(1)
            current_test = TestCase(name=test_name)
            continue

        # ── Task started ──
        if 'ℹ️  TASK STARTED' in content:
            task_text = content.split('TASK STARTED', 1)[1].lstrip(' :').strip()
            new_agent = Agent(task=task_text)

            if not current_test:
                # No test case context yet, create one from filename
                current_test = TestCase(name="unknown")

            if not current_test.main_agent:
                # First agent = main agent
                new_agent.is_main = True
                current_test.main_agent = new_agent
                agent_stack = [new_agent]
            else:
                # Sub-agent
                # Use the task from the microbot_sub command if available
                if pending_sub_agent_step and pending_sub_agent_step.sub_agent_task:
                    new_agent.task = pending_sub_agent_step.sub_agent_task
                elif task_text:
                    new_agent.task = task_text

                sub_idx = len(current_test.sub_agents)
                current_test.sub_agents.append(new_agent)

                # Link the parent step to this sub-agent
                if pending_sub_agent_step:
                    pending_sub_agent_step.sub_agent_index = sub_idx
                    pending_sub_agent_step = None

                agent_stack.append(new_agent)

            current_step = None
            current_field = None
            continue

        # ── Task completed ──
        if '🔚 TASK COMPLETED' in content:
            agent = current_agent()
            if agent:
                agent.completed = True
            current_field = None  # Stop accumulating text
            continue

        # ── Sub-agent completed message ──
        if 'Sub-agent completed successfully with output:' in content:
            # Pop sub-agent from stack
            if len(agent_stack) > 1:
                agent_stack.pop()
            current_step = None
            current_field = None
            continue

        # ── Sub-agent failed ──
        if level == 'ERROR' and 'Sub-agent failed' in content:
            agent = current_agent()
            if agent and not agent.is_main:
                agent.max_iterations_reached = True
                agent.completed = False
            # Pop sub-agent from stack
            if len(agent_stack) > 1:
                agent_stack.pop()
            current_step = None
            current_field = None
            continue

        # ── Max iterations reached ──
        if level == 'ERROR' and 'Max iterations' in content:
            agent = current_agent()
            if agent:
                agent.max_iterations_reached = True
            continue

        # ── Step boundary ──
        step_match = re.search(r'-+ Step-(\d+) -+', content)
        if step_match:
            step_num = int(step_match.group(1))
            current_step = Step(number=step_num)
            agent = current_agent()
            if agent:
                agent.steps.append(current_step)
            current_field = None
            continue

        # ── LLM final thoughts ──
        if '💭  LLM final thoughts:' in content:
            text = content.split('💭  LLM final thoughts:', 1)[1].strip()
            agent = current_agent()
            if agent:
                agent.final_thoughts = text
            current_field = 'final_thoughts'
            continue

        # ── LLM thoughts ──
        if '💭  LLM thoughts:' in content:
            text = content.split('💭  LLM thoughts:', 1)[1].strip()
            if current_step:
                current_step.thought = text
            current_field = 'thought'
            continue

        # ── LLM tool call ──
        if '➡️  LLM tool call :' in content:
            cmd = content.split('➡️  LLM tool call :', 1)[1].strip()
            # Remove surrounding quotes if present
            if cmd.startswith('"') and cmd.endswith('"'):
                cmd = cmd[1:-1]
            if current_step:
                current_step.command = cmd
                if 'microbot_sub' in cmd:
                    current_step.is_sub_agent_call = True
                    current_step.sub_agent_task = extract_task_from_microbot_sub(cmd)
                    pending_sub_agent_step = current_step
            current_field = 'command'
            continue

        # ── Command output ──
        if '⬅️  Command output:' in content:
            text = content.split('⬅️  Command output:', 1)[1].strip()
            if current_step:
                current_step.output = text
            current_field = 'output'
            continue

        # ── Dangerous command blocked ──
        if '⚠️  Dangerous command detected' in content:
            if current_step:
                current_step.is_blocked = True
                current_step.blocked_reason = content
            current_field = 'blocked'
            continue

        # ── REASON / ALTERNATIVE for blocked commands ──
        if current_field == 'blocked' and current_step:
            if content.startswith('REASON:'):
                current_step.blocked_reason = content
            elif content.startswith('ALTERNATIVE:'):
                current_step.blocked_alternative = content
            continue

        # ── Invoking MicroBotSubAgent ──
        if 'Invoking MicroBotSubAgent with task:' in content:
            # This is just a log message; the sub-agent TASK STARTED follows
            continue

        # ── Multi-line continuation for known fields ──
        if current_field == 'output' and current_step:
            if current_step.output:
                current_step.output += '\n' + content
            else:
                current_step.output = content
            continue

        if current_field == 'thought' and current_step:
            if current_step.thought:
                current_step.thought += '\n' + content
            else:
                current_step.thought = content
            continue

        if current_field == 'command' and current_step:
            if current_step.command:
                current_step.command += '\n' + content
            else:
                current_step.command = content
            continue

        if current_field == 'final_thoughts':
            agent = current_agent()
            if agent:
                if agent.final_thoughts:
                    agent.final_thoughts += '\n' + content
                else:
                    agent.final_thoughts = content
            continue

    # Finalize last test case
    finalize_test_case()

    return test_cases


# ─────────────────────────── Markdown Generation ───────────────────────────


def escape_md(text: str) -> str:
    """Escape text for markdown display (minimal escaping for code blocks)."""
    return text


def truncate_text(text: str, max_lines: int = 200) -> str:
    """Truncate text if it exceeds max_lines."""
    lines = text.split('\n')
    if len(lines) > max_lines:
        return '\n'.join(lines[:max_lines]) + f'\n\n... ({len(lines) - max_lines} more lines truncated)'
    return text


def generate_step_md(step: Step, sub_agent_filename: str = "") -> str:
    """Generate markdown for a single step as a collapsible details section."""
    status = "🚫 Blocked" if step.is_blocked else ""
    if step.is_sub_agent_call:
        status = "🤖 Sub-Agent Call"

    summary = f"Step {step.number}"
    if status:
        summary += f" - {status}"

    # Build brief description from the thought (first sentence)
    if step.thought:
        first_line = step.thought.split('\n')[0]
        if len(first_line) > 120:
            first_line = first_line[:117] + "..."
        summary += f": {first_line}"

    md = f"<details>\n<summary><strong>{summary}</strong></summary>\n\n"

    # Thought section
    if step.thought:
        md += "### 💭 Thought\n\n"
        md += f"{step.thought}\n\n"

    # Blocked command warning
    if step.is_blocked:
        md += "### ⚠️ Command Blocked\n\n"
        if step.blocked_reason:
            md += f"> {step.blocked_reason}\n"
        if step.blocked_alternative:
            md += f"> {step.blocked_alternative}\n"
        md += "\n"

    # Command section
    if step.command:
        md += "### ➡️ Command\n\n"
        if step.is_sub_agent_call:
            md += "**Sub-agent invocation:**\n\n"
            if sub_agent_filename:
                md += f"📎 **[View Sub-Agent Trajectory]({sub_agent_filename})**\n\n"
            # Show the task
            if step.sub_agent_task:
                md += "<details>\n<summary>Sub-agent task description</summary>\n\n"
                md += f"```\n{step.sub_agent_task}\n```\n\n"
                md += "</details>\n\n"
        else:
            md += f"```bash\n{step.command}\n```\n\n"

    # Output section
    if step.output:
        md += "### ⬅️ Output\n\n"
        output_text = truncate_text(step.output)
        md += f"```\n{output_text}\n```\n\n"

    md += "</details>\n\n"
    return md


def generate_main_agent_md(test_case: TestCase) -> str:
    """Generate the main agent markdown file content."""
    md = f"# 🤖 Main Agent Trajectory: {test_case.name}\n\n"

    if test_case.main_agent and test_case.main_agent.task:
        md += "## Task\n\n"
        task_text = test_case.main_agent.task
        if len(task_text) > 500:
            md += f"<details>\n<summary>Full task description</summary>\n\n{task_text}\n\n</details>\n\n"
        else:
            md += f"{task_text}\n\n"

    md += "---\n\n"
    md += "## Steps\n\n"

    if test_case.main_agent:
        agent = test_case.main_agent
        for step in agent.steps:
            sub_agent_file = ""
            if step.is_sub_agent_call and step.sub_agent_index >= 0:
                sub_agent_file = f"sub_agent_{step.sub_agent_index + 1}.md"
            md += generate_step_md(step, sub_agent_filename=sub_agent_file)

        md += "---\n\n"

        # Summary
        if agent.completed:
            md += "## ✅ Task Completed\n\n"
            if agent.final_thoughts:
                md += f"{agent.final_thoughts}\n\n"
        elif agent.max_iterations_reached:
            md += "## ❌ Max Iterations Reached\n\n"
            md += "The agent did not complete the task within the maximum allowed iterations.\n\n"

        # Sub-agent index
        if test_case.sub_agents:
            md += "## 📋 Sub-Agents\n\n"
            md += "| # | Task | Status | Link |\n"
            md += "|---|------|--------|------|\n"
            for i, sub in enumerate(test_case.sub_agents):
                clean = clean_task_text(sub.task)
                first_line = clean.split('\n')[0]
                task_summary = first_line[:80] + "..." if len(first_line) > 80 else first_line
                task_summary = task_summary.replace('|', '\\|')
                status = "✅ Completed" if sub.completed else "❌ Failed"
                link = f"[sub_agent_{i + 1}.md](sub_agent_{i + 1}.md)"
                md += f"| {i + 1} | {task_summary} | {status} | {link} |\n"
            md += "\n"

    return md


def clean_task_text(task: str) -> str:
    """Clean up a task string: remove microbot_sub prefix, escaped quotes, etc."""
    text = task.strip()
    # Remove microbot_sub --task "..." wrapper if present
    if text.startswith('microbot_sub'):
        match = re.search(r'--task\s+["\'](.+)', text, re.DOTALL)
        if match:
            text = match.group(1)
            # Remove trailing quote + flags
            text = re.sub(r'["\']\s*--(?:iterations|timeout).*$', '', text, flags=re.DOTALL)
            text = text.strip().strip('"').strip("'").strip()
    # Unescape
    text = text.replace('\\"', '"').replace('\\n', '\n').replace("\\'" , "'")
    return text


def generate_sub_agent_md(sub_agent: Agent, index: int, test_case_name: str) -> str:
    """Generate a sub-agent markdown file content."""
    # Clean and use the first line of the task as heading
    clean_task = clean_task_text(sub_agent.task)
    task_heading = clean_task.split('\n')[0] if clean_task else f"Sub-Agent {index + 1}"
    if len(task_heading) > 150:
        task_heading = task_heading[:147] + "..."

    md = f"# {task_heading}\n\n"
    md += "**Parent:** [Main Agent](main_agent.md) | "
    md += f"**Test Case:** {test_case_name}\n\n"

    if clean_task and '\n' in clean_task:
        md += "<details>\n<summary>Full task description</summary>\n\n"
        md += f"```\n{clean_task}\n```\n\n"
        md += "</details>\n\n"

    md += "---\n\n"
    md += "## Steps\n\n"

    for step in sub_agent.steps:
        md += generate_step_md(step)

    md += "---\n\n"

    # Summary
    if sub_agent.completed:
        md += "## ✅ Task Completed\n\n"
        if sub_agent.final_thoughts:
            md += f"{sub_agent.final_thoughts}\n\n"
    elif sub_agent.max_iterations_reached:
        md += "## ❌ Max Iterations Reached\n\n"
        md += "The sub-agent did not complete the task within the maximum allowed iterations.\n\n"

    return md


# ─────────────────────────── Main ───────────────────────────


def parse_and_generate(log_path: str, output_base_dir: str = None):
    """
    Parse an info.log file and generate markdown trajectory files.

    Args:
        log_path: Path to the info.log file
        output_base_dir: Base directory for output. If None, uses the log file's directory.
    """
    if not os.path.isfile(log_path):
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)

    # Derive test case name from filename
    basename = os.path.basename(log_path)
    if basename.endswith('_info.log'):
        default_test_name = basename[:-len('_info.log')]
    elif basename.endswith('.log'):
        default_test_name = basename[:-len('.log')]
    else:
        default_test_name = os.path.splitext(basename)[0]

    if output_base_dir is None:
        output_base_dir = os.path.dirname(os.path.abspath(log_path))

    print(f"Parsing log file: {log_path}")

    # Parse
    entries = parse_log_entries(log_path)
    print(f"  Parsed {len(entries)} log entries")

    test_cases = build_test_cases(entries)
    print(f"  Found {len(test_cases)} test case(s)")

    if not test_cases:
        # If no test case boundaries found, create a single test case
        print("  No test case boundaries found, treating entire log as one test case")
        tc = TestCase(name=default_test_name)
        # Re-parse with a dummy test case
        test_cases = _build_single_test_case(entries, default_test_name)

    for tc in test_cases:
        # Create output directory
        trajectory_dir = os.path.join(output_base_dir, f"{tc.name}_trajectory")
        os.makedirs(trajectory_dir, exist_ok=True)
        print(f"\n  Test case: {tc.name}")
        print(f"  Output directory: {trajectory_dir}")

        # Generate main agent markdown
        main_md = generate_main_agent_md(tc)
        main_path = os.path.join(trajectory_dir, "main_agent.md")
        with open(main_path, 'w', encoding='utf-8') as f:
            f.write(main_md)
        main_steps = len(tc.main_agent.steps) if tc.main_agent else 0
        print(f"  Created: main_agent.md ({main_steps} steps)")

        # Generate sub-agent markdowns
        for i, sub in enumerate(tc.sub_agents):
            sub_md = generate_sub_agent_md(sub, i, tc.name)
            sub_path = os.path.join(trajectory_dir, f"sub_agent_{i + 1}.md")
            with open(sub_path, 'w', encoding='utf-8') as f:
                f.write(sub_md)
            print(f"  Created: sub_agent_{i + 1}.md ({len(sub.steps)} steps)")

    print(f"\nDone! Generated trajectory files for {len(test_cases)} test case(s).")
    return test_cases


def _build_single_test_case(entries: List[dict], name: str) -> List[TestCase]:
    """
    Build a single test case when no test directory boundaries are found.
    Injects a fake test case boundary entry and delegates to build_test_cases.
    """
    fake_boundary = {
        'timestamp': '2000-01-01 00:00:00,000',
        'level': 'INFO',
        'content': f'Test directory set up at: /fake/{name}',
        'line_num': 0,
    }
    return build_test_cases([fake_boundary] + entries)


def main():
    if len(sys.argv) < 2:
        print("Usage: python multi_agent_log_parser.py <test_case>_info.log [output_dir]")
        print("\nParses an info.log file and generates markdown trajectory files.")
        print("The log file should be named as <test_case>_info.log.")
        print("A directory <test_case>_trajectory will be created with all markdown files.")
        sys.exit(1)

    log_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    parse_and_generate(log_path, output_dir)


if __name__ == '__main__':
    main()
