# Microbots

MicroBots is a lightweight, extensible AI agent for code comprehension and controlled file edits. It integrates cleanly 
into automation pipelines, mounting a target directory with explicit read-only or read/write modes so LLMs can safely 
inspect, refactor, or generate files with least‑privilege access.

```py
myWritingAgent = WritingBot(
    model="azure-openai/gpt5",
    folder_to_mount=str("myReactApp"),
)
data = myWritingAgent.run("when doing npm run build, I get an error. Fix the error and make sure the build is successful.", timeout_in_seconds=600)
print(data.results)
```

## How to install

```bash
pip install microbots
```


## Pre-requisites

- Docker
- AI LLM Provider and API Key (OpenAI, Azure OpenAI etc.,)


## LLM Support
    
OpenAI Models

## Bots & Usage Examples

Pre-requisite for the below example code of Bots:   
From the root of your application, Create a folder called  `code` inside which clone the repo `https://github.com/swe-agent/test-repo/`. Now run the code


### ReadingBot


```py
from microbots import ReadingBot

myBot = ReadingBot(
    model="azure-openai/mini-swe-agent-gpt5",
    folder_to_mount="code"
)

runResult = myBot.run("When I am running missing_colon.py I am getting SyntaxError: invalid syntax. Find the error and explain me what is the error", timeout_in_seconds=600)
print(runResult)

```

The `ReadingBot` mounts the `code` folder in `READ_ONLY` mode securely for llm to read and understand the code base.


### WritingBot

Pre-requisite for the example code: 
From the root the application, Create a folder called  `code` inside which clone the repo `https://github.com/swe-agent/test-repo/`. Now run the code

```py
from microbots import WritingBot

myBot = WritingBot(
    model="azure-openai/mini-swe-agent-gpt5",
    folder_to_mount="code"
)

myBot.run("When I am running missing_colon.py I am getting SyntaxError: invalid syntax. Fix the error and make sure the code runs without any errors.", timeout_in_seconds=600)
```

The `WritingBot` mounts the `code` folder in `READ_WRITE` mode securely for llm to read and edit the code base.