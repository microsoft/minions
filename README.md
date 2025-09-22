# 🤖 Microbots

MicroBots is a lightweight, extensible AI agent for code comprehension and controlled file edits. It integrates cleanly 
into automation pipelines, mounting a target directory with explicit read-only or read/write modes so LLMs can safely 
inspect, refactor, or generate files with least‑privilege access.


```py
from microbots import WritingBot

myWritingBot = WritingBot(
    model="azure-openai/my-gpt5", # model format : <provider/deployment_model_name>
    folder_to_mount=str("myReactApp"),
)

data = myWritingBot.run("""when doing npm run build, I get an error. 
Fix the error and make sure the build is successful.""", timeout_in_seconds=600)
print(data.results)
```

## ⚠️ Caution 

As of now the library is at `beta`. Use it with caution.
We are adding more features and improving the library.

## 🚀 How to install

### Pre-requisites

- Docker
- AI LLM Provider and API Key

### Install Microbots

```bash
pip install microbots
```


## ✨LLM Support
    
Azure OpenAI Models - Add the below environment variables in a `.env` file in the root of your application

```env
OPEN_AI_END_POINT=XXXXXXXXXXXXXXXXXXXXXXXXXX
OPEN_AI_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

## 🤖 Bots & Usage Examples

Pre-requisite for the below example code of Bots:   
From the root of your application, Create a folder called  `code` inside which clone the repo `https://github.com/swe-agent/test-repo/`. Now run the code


### 📖 ReadingBot


```py
from microbots import ReadingBot

myBot = ReadingBot(
    model="azure-openai/my-gpt5",
    folder_to_mount="code"
)

runResult = myBot.run("When I am running missing_colon.py I am getting SyntaxError: invalid syntax. Find the error and explain me what is the error", timeout_in_seconds=600)
print(runResult)

```

The `ReadingBot` will read the files inside `code` folder and will extract information 


### ✍️ WritingBot

Pre-requisite for the example code: 
From the root the application, Create a folder called  `code` inside which clone the repo `https://github.com/swe-agent/test-repo/`. Now run the code

```py
from microbots import WritingBot

myBot = WritingBot(
    model="azure-openai/my-gpt5",
    folder_to_mount="code"
)

myBot.run("When I am running missing_colon.py I am getting SyntaxError: invalid syntax. Fix the error and make sure the code runs without any errors.", timeout_in_seconds=600)
```

## ⚙️ How it works


![Overall Architecture Image](./docs/images/overall_architecture.png)

The MicroBots create a container environment to install tools and inspect codes on specific access based on the Bot user uses to interact with the codebase. By restricting the permissions to read-only or read/write, it ensures that the AI agents operate within defined boundaries, enhancing security and control over code modifications as well as protecting the local environment.