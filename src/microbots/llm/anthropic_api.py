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
        additional_tools: Optional[List] = None,
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
        additional_tools : Optional[List]
            Tool objects passed from MicroBot. Any tools exposing a native
            Anthropic schema via ``to_dict()`` are forwarded to the API.
        """
        self.ai_client = Anthropic(
            api_key=api_key,
            base_url=endpoint
        )
        self.deployment_name = deployment_name
        self.system_prompt = system_prompt
        self.messages = []

        # Preserve tool instances as provided and extract native Anthropic schemas.
        tools = additional_tools or []
        upgraded = self.upgrade_tools(tools)
        if additional_tools is not None:
            additional_tools[:] = upgraded
        self._tool_dicts = [
            t.to_dict() for t in upgraded
            if callable(getattr(t, "to_dict", None))
        ]
        self._pending_tool_response = None

        # Set these values here. This logic will be handled in the parent class.
        self.max_retries = max_retries
        self.retries = 0

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    def _call_api(self) -> object:
        """Call the Anthropic messages API, including tool definitions when present."""
        kwargs = dict(
            model=self.deployment_name,
            system=self.system_prompt,
            messages=self.messages,
            max_tokens=4096,
        )

        if self._tool_dicts:
            kwargs["tools"] = self._tool_dicts

        return self.ai_client.messages.create(**kwargs)

    def _append_tool_result(self, response, result_text: str) -> None:
        """Append the assistant tool_use turn and the corresponding tool_result user turn.

        Called when the caller provides the tool execution result via
        the next ``ask()`` call.
        """
        assistant_content = [block.model_dump() for block in response.content]
        self.messages.append({"role": "assistant", "content": assistant_content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
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

        if self._pending_tool_response:
            # Previous response was tool_use — format this message as tool results.
            self._append_tool_result(self._pending_tool_response, message)
            self._pending_tool_response = None
        else:
            self.messages.append({"role": "user", "content": message})

        valid = False
        while not valid:
            response = self._call_api()

            if response.stop_reason == "tool_use":
                # Return tool call info as an LLMAskResponse so the
                # caller (MicroBot.run) can dispatch the tool.
                self._pending_tool_response = response

                thoughts = ""
                for block in response.content:
                    if block.type == "text":
                        thoughts = block.text
                        break

                tool_calls = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_calls.append({
                            "name": block.name,
                            "id": block.id,
                            "input": block.input,
                        })

                command = json.dumps({"native_tool_calls": tool_calls})
                return LLMAskResponse(task_done=False, thoughts=thoughts, command=command)

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
