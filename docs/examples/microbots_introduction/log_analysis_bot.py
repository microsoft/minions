from microbots import LogAnalysisBot

my_bot = LogAnalysisBot(
    model="azure-openai/my-gpt-5",
    folder_to_mount="code",
)

result = my_bot.run(
    file_name="code/build.log",
    timeout_in_seconds=600,
)
print(result.result)
