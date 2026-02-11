import json
import os
import re
from dataclasses import asdict
from logging import getLogger
from typing import Optional, cast

from dotenv import load_dotenv
from anthropic import Anthropic
from anthropic.types.beta import BetaContextManagementConfigParam
from microbots.llm.llm import LLMAskResponse, LLMInterface
from microbots.tools.external_tool import ExternalTool

logger = getLogger(__name__)

load_dotenv()

endpoint = os.getenv("ANTHROPIC_END_POINT")
deployment_name = os.getenv("ANTHROPIC_DEPLOYMENT_NAME")
api_key = os.getenv("ANTHROPIC_API_KEY")


class AnthropicApi(LLMInterface):

    def __init__(
        self,
        system_prompt,
        deployment_name=deployment_name,
        max_retries=3,
        external_tools: Optional[list[ExternalTool]] = None,
        context_management: Optional[dict] = None,
    ):
        self.ai_client = Anthropic(
            api_key=api_key,
            base_url=endpoint
        )
        self.deployment_name = deployment_name
        self.external_tools = external_tools or []
        self.context_management = context_management
        self.system_prompt = system_prompt
        self.messages = []

        # Set these values here. This logic will be handled in the parent class.
        self.max_retries = max_retries
        self.retries = 0


    def ask(self, message) -> LLMAskResponse:
        if self.external_tools:
            return self._ask_with_tools(message)
        return self._ask_simple(message)

    def clear_history(self):
        self.messages = []
        return True


    def _ask_simple(self, message) -> LLMAskResponse:
        self.retries = 0

        self.messages.append({"role": "user", "content": message})

        valid = False
        while not valid:
            response = self.ai_client.messages.create(
                model=self.deployment_name,
                system=self.system_prompt,
                messages=self.messages,
                max_tokens=4096,
            )

            response_text = response.content[0].text if response.content else ""
            logger.debug("Raw Anthropic response (first 500 chars): %s", response_text[:500])

            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            valid, askResponse = self._validate_llm_response(response=response_text)

        self.messages.append({"role": "assistant", "content": json.dumps(asdict(askResponse))})

        return askResponse


    def _ask_with_tools(self, message) -> LLMAskResponse:
        """Ask with external tools, using the SDK tool_runner for the loop."""
        self.retries = 0

        self.messages.append({"role": "user", "content": message})

        # Collect beta headers from tools
        betas = self._collect_betas()

        # Snapshot message count so we can roll back on validation retry
        messages_snapshot = len(self.messages)

        valid = False
        while not valid:
            # Roll back any messages appended by a previous failed iteration
            del self.messages[messages_snapshot:]

            runner_kwargs: dict = {
                "model": self.deployment_name,
                "system": self.system_prompt,
                "messages": list(self.messages),  # copy to avoid mutation
                "tools": self.external_tools,
                "max_tokens": 4096,
            }
            if betas:
                runner_kwargs["betas"] = betas
            if self.context_management:
                runner_kwargs["context_management"] = cast(
                    BetaContextManagementConfigParam, self.context_management
                )

            runner = self.ai_client.beta.messages.tool_runner(**runner_kwargs)

            # Consume the runner, tracking messages for conversation history
            last_message = None
            for response_message in runner:
                last_message = response_message

                # Build assistant content for our message history
                assistant_content = self._build_message_content(response_message)
                if assistant_content:
                    self.messages.append({"role": "assistant", "content": assistant_content})

                # Get tool results (runner caches, so tools execute only once)
                tool_response = runner.generate_tool_call_response()
                if tool_response and tool_response.get("content"):
                    self.messages.append(tool_response)

            # Extract final text
            response_text = self._extract_text(last_message) if last_message else ""
            logger.debug("Raw Anthropic response (first 500 chars): %s", response_text[:500])

            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

            valid, askResponse = self._validate_llm_response(response=response_text)

        # Find and replace the last assistant message with the validated response
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].get("role") == "assistant":
                self.messages[i] = {"role": "assistant", "content": json.dumps(asdict(askResponse))}
                break
        else:
            self.messages.append({"role": "assistant", "content": json.dumps(asdict(askResponse))})

        return askResponse


    def _collect_betas(self) -> list[str]:
        """Collect unique beta headers from all external tools."""
        betas = set()
        for tool in self.external_tools:
            if hasattr(tool, "beta_header") and tool.beta_header:
                betas.add(tool.beta_header)
        return list(betas) if betas else []

    @staticmethod
    def _build_message_content(response_message) -> list:
        """Convert an API response message into a serialisable content list."""
        content = []
        for block in response_message.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                content.append({"type": "text", "text": block.text})
            elif block_type == "tool_use":
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
            else:
                # server_tool_use or other block types
                block_dict = block.model_dump() if hasattr(block, "model_dump") else block
                content.append(block_dict)
        return content

    @staticmethod
    def _extract_text(response) -> str:
        """Extract all text content from an API response."""
        if response is None:
            return ""
        texts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                texts.append(block.text)
        return "\n".join(texts) if texts else ""
