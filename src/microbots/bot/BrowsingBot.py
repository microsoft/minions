from typing import Optional

from microbots.constants import PermissionLabels
from microbots.MicroBot import BotType, MicroBot, BotRunResult
from microbots.environment.Environment import Environment
from microbots.tools.tool import Tool, parse_tool_definition, setup_tools


BROWSER_USE_TOOL = parse_tool_definition("browser-use.yaml")


class BrowsingBot(MicroBot):

    def __init__(
        self,
        model: str,
        environment: Optional[Environment] = None,
        additional_tools: Optional[list[Tool]] = [],
    ):
        # validate init values before assigning
        bot_type = BotType.BROWSING_BOT
        system_prompt = """
        You search the web to gather information about a topic.
        """

        super().__init__(
            bot_type=bot_type,
            model=model,
            system_prompt=system_prompt,
            environment=environment,
            additional_tools=additional_tools + [BROWSER_USE_TOOL],
        )

    def run(self, task, max_iterations=20, timeout_in_seconds=200) -> BotRunResult:
        setup_tools(self.environment, self.additional_tools)
        # browser-use will run inside the docker. So, single command to env should be sufficient
        browser_output = self.environment.execute(f"browser '{task}'", timeout=timeout_in_seconds)
        if browser_output.return_code != 0:
            return f"Failed to run browser command. Error: {browser_output.stderr}"

        browser_stdout = browser_output.stdout
        # print("Browser stdout:", browser_stdout)
        # final_result = browser_stdout.split("Final result:")[-1].strip() if "Final result:" in browser_stdout else browser_stdout.strip()
        final_result = browser_stdout["Final result:"] if "Final result:" in browser_stdout else browser_stdout

        return BotRunResult(
            status=browser_output.return_code == 0,
            result=final_result,
            error=browser_output.stderr if browser_output.return_code != 0 else None,
        )

