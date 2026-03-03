import json
import os
import re
from dataclasses import asdict
from logging import getLogger
from typing import List, Optional

from dotenv import load_dotenv
from anthropic import Anthropic
from microbots.llm.llm import LLMAskResponse, LLMInterface

logger = getLogger(__name__)

load_dotenv()

endpoint = os.getenv("ANTHROPIC_END_POINT")
deployment_name = os.getenv("ANTHROPIC_DEPLOYMENT_NAME")
api_key = os.getenv("ANTHROPIC_API_KEY")



class AnthropicApi(LLMInterface):

    def __init__(
        self,
        system_prompt: str,
        deployment_name: str = deployment_name,
        max_retries: int = 3,
        native_tools: Optional[List] = None,
    ):
        """
        Parameters
        ----------
        system_prompt : str
            System prompt for the LLM.
        deployment_name : str
            The Anthropic model deployment name.
        max_retries : int
            Maximum number of retries for invalid LLM responses.
        native_tools : Optional[List]
            Anthropic-native tool objects (e.g. ``AnthropicMemoryTool``) that
            have both a ``to_dict()`` and a ``call()`` method.  These are passed
            directly to the API and their tool-use blocks are dispatched here
            before the JSON response is returned to the caller.
        """
        self.ai_client = Anthropic(
            api_key=api_key,
            base_url=endpoint
        )
        self.deployment_name = deployment_name
        self.system_prompt = system_prompt
        self.messages = []
        self.native_tools = native_tools or []
        # Cache tool dicts once so _call_api and _dispatch_tool_use don't
        # re-serialise on every invocation (important when multiple native
        # tools are registered, e.g. memory + bash).
        self._native_tool_dicts = [t.to_dict() for t in self.native_tools]
        self._native_tools_by_name = {d["name"]: t for d, t in zip(self._native_tool_dicts, self.native_tools)}

        # Set these values here. This logic will be handled in the parent class.
        self.max_retries = max_retries
        self.retries = 0

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _call_api(self) -> object:
        """Call the Anthropic messages API, including native tools when present."""
        kwargs = dict(
            model=self.deployment_name,
            system=self.system_prompt,
            messages=self.messages,
            max_tokens=4096,
        )

        if self.native_tools:
            kwargs["tools"] = self._native_tool_dicts

        return self.ai_client.messages.create(**kwargs)

    def _dispatch_tool_use(self, response) -> None:
        """Handle a tool_use response: execute each tool call and append results.

        Mutates ``self.messages`` in place — appends the assistant turn (with
        all content blocks) and the corresponding tool_result user turn.
        """
        # Append the full assistant message as-is (content is a list of blocks)
        assistant_content = [block.model_dump() for block in response.content]
        self.messages.append({"role": "assistant", "content": assistant_content})

        # Build tool_result entries for every tool_use block
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            # Find the matching native tool by name
            tool = self._native_tools_by_name.get(block.name)
            if tool is None:
                result_text = f"Error: unknown tool '{block.name}'"
                logger.error("Received tool_use for unknown tool: %s", block.name)
            else:
                try:
                    result_text = tool.call(block.input)
                    logger.info(
                        "🧠 Native tool '%s' executed. Result (first 200 chars): %s",
                        block.name,
                        str(result_text)[:200],
                    )
                except Exception as exc:
                    result_text = f"Error executing tool '{block.name}': {exc}"
                    logger.error("Native tool '%s' raised: %s", block.name, exc)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(result_text),
            })

        self.messages.append({"role": "user", "content": tool_results})

    # ---------------------------------------------------------------------- #
    # Public interface
    # ---------------------------------------------------------------------- #

    def ask(self, message: str) -> LLMAskResponse:
        self.retries = 0  # reset retries for each ask. Handled in parent class.

        self.messages.append({"role": "user", "content": message})

        valid = False
        while not valid:
            response = self._call_api()

            # Dispatch any tool_use rounds before looking for a JSON response.
            # The model may call the memory tool multiple times before producing
            # its final JSON command.
            while response.stop_reason == "tool_use":
                self._dispatch_tool_use(response)
                response = self._call_api()

            # Extract text content from the final response
            response_text = ""
            for block in response.content:
                if block.type == "text":
                    response_text = block.text
                    break

            logger.debug("Raw Anthropic response (first 500 chars): %s", response_text[:500])

            # Try to extract JSON if wrapped in markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            valid, askResponse = self._validate_llm_response(response=response_text)

        self.messages.append({"role": "assistant", "content": json.dumps(asdict(askResponse))})

        return askResponse

    def clear_history(self):
        self.messages = []
        return True
