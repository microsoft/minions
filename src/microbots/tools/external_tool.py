"""
ExternalTool — abstract base for LLM-native tools that run *outside* the Docker sandbox.

These tools are invoked directly by the LLM via the provider's tool-use protocol
(e.g., Anthropic tool_use blocks) rather than being installed inside a container.

Hierarchy:
    Tool (ABC)
    └── ExternalTool (Tool)
            └── AnthropicMemoryTool
"""

from abc import abstractmethod

from microbots.tools.tool import Tool


class ExternalTool(Tool):
    """
    Abstract base class for LLM-native tools.

    Subclasses must provide ``get_tool_definition()``.
    Subclasses *may* override ``execute()`` for direct invocation;
    SDK-dispatched tools (e.g., via ``tool_runner``) can skip it.
    """

    @abstractmethod
    def get_tool_definition(self) -> dict:
        """
        Return the tool definition dict to send to the LLM provider.

        The format depends on the provider — e.g., for Anthropic it
        should match the beta tool schema.
        """
        ...

    def execute(self, tool_input: dict) -> str:
        """
        Execute the tool with the given input and return the result as a string.

        Override this in subclasses that support direct invocation.
        SDK-dispatched tools (where the provider's tool_runner handles
        dispatch) do not need to override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support direct execute(). "
            "This tool is dispatched by the provider SDK's tool_runner."
        )
