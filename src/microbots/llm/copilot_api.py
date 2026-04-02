import asyncio
import json
import os
import threading
from dataclasses import asdict
from logging import getLogger

from copilot import CopilotClient, PermissionHandler
from copilot.types import SubprocessConfig
from microbots.llm.llm import LLMAskResponse, LLMInterface
from microbots.utils.copilot_auth import get_copilot_token

logger = getLogger(__name__)


class CopilotApi(LLMInterface):

    def __init__(self, system_prompt, model_name, max_retries=3, github_token=None):
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.messages = []

        self.max_retries = max_retries
        self.retries = 0

        # Resolve GitHub token: explicit > env var > ~/.copilot/config.json > SDK default
        self._github_token = github_token or os.environ.get("GITHUB_TOKEN") or get_copilot_token()

        # Persistent event loop in a daemon thread for async-sync bridging.
        # The Copilot SDK is async-native; MicroBot's LLMInterface is sync.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True
        )
        self._thread.start()
        self._closed = False

        config = SubprocessConfig(github_token=self._github_token) if self._github_token else SubprocessConfig()
        self._client = CopilotClient(config)
        self._session = None
        self._run_async(self._start())

    async def _start(self):
        await self._client.start()
        await self._create_session()

    async def _create_session(self):
        self._session = await self._client.create_session(
            model=self.model_name,
            on_permission_request=PermissionHandler.approve_all,
            system_message={"content": self.system_prompt},
            infinite_sessions={"enabled": False},
        )

    def _run_async(self, coro):
        """Submit an async coroutine to the background loop and block until done."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def _send_and_collect(self, message):
        """Send a message and wait for the assistant's complete response."""
        response_event = await self._session.send_and_wait(message, timeout=300.0)
        if response_event and response_event.data and response_event.data.content:
            return response_event.data.content
        return ""

    def ask(self, message) -> LLMAskResponse:
        self.retries = 0

        self.messages.append({"role": "user", "content": message})

        valid = False
        while not valid:
            response_text = self._run_async(self._send_and_collect(message))
            logger.debug(
                "Raw Copilot response (first 500 chars): %s",
                response_text[:500],
            )

            # Try to extract JSON if wrapped in markdown code blocks
            import re
            json_match = re.search(
                r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL
            )
            if json_match:
                response_text = json_match.group(1)

            valid, askResponse = self._validate_llm_response(
                response=response_text
            )

        self.messages.append(
            {"role": "assistant", "content": json.dumps(asdict(askResponse))}
        )
        return askResponse

    def clear_history(self):
        self.messages = []
        self._run_async(self._recreate_session())
        return True

    async def _recreate_session(self):
        if self._session:
            await self._session.disconnect()
        await self._create_session()

    def close(self):
        """Stop the Copilot client and shut down the background event loop."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._session:
                self._run_async(self._session.disconnect())
            self._run_async(self._client.stop())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
