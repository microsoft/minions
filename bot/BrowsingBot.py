from typing import Optional

from constants import PermissionLabels
from MicroBot import BotType, MicroBot, system_prompt_common
from tools.tool import Tool, parse_tool_definition


BROWSER_USE_TOOL = parse_tool_definition("browser-use.yaml")


class BrowsingBot(MicroBot):

    def __init__(
        self,
        model: str,
        environment: Optional[any] = None,
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
