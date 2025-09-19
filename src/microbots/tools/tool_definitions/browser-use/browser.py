#!/usr/bin/env python3

import asyncio
import sys
from pprint import pprint

from dotenv import load_dotenv
load_dotenv()

from browser_use import Agent, AgentHistoryList, Browser, ChatAzureOpenAI


async def main(args: list[str]) -> int:
    if len(args) > 1:
        print("browse allows only one query at a time.")
        return 1

    if not args:
        print("Usage: browse 'query_to_search'")
        return 1

    what_to_browse = args[0]

    browser = Browser(
        headless=True
        )

    agent = Agent(
        task=what_to_browse,
        browser=browser,
        llm=ChatAzureOpenAI(model="gpt-4.1"),
        use_vision=False,
    )
    history: AgentHistoryList = await agent.run()
    print("Final Result:")
    pprint(history.final_result(), indent=4)


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))
