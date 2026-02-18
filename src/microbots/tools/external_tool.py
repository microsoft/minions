import logging
from dataclasses import dataclass
from microbots.environment import Environment
from microbots.tools.tool import ToolAbstract, TOOLTYPE
import os

logger = logging.getLogger(" 🔧 ExternalTool")

@dataclass
class ExternalTool(ToolAbstract):
    """
    Tools that are executed in the current environment (Host) and are directly accessible by the LLM without any installation inside the Docker sandbox.
    Examples include web search tools, retrieval tools, etc.
    """
    pass

    def __init__(self):
        super().__init__()
        self.tool_type = TOOLTYPE.EXTERNAL

    def _install_external_tool(self, env: Environment):
        logger.debug("Installing external tool: %s", self.name)
        for command in self.install_commands:
            output = os.system(command)
            logger.debug("External tool install command: %s", command)
            if output != 0:
                logger.error(
                    "❌ Failed to install external tool: %s with command: %s",
                    self.name,
                    command,
                )
                raise RuntimeError(
                    f"Failed to install external tool {self.name} with command {command}. Output: {output}"
                )
        logger.info("✅ Successfully installed external tool: %s", self.name)

    def _verify_env_variable(self, env: Environment, env_variable: str):
        # For external tools, env variables will not be set. Instead this function will simply
        # check if the env variable is present in the host environment and raise an error if not.
        if env_variable not in os.environ:
            logger.error(
                "❌ Environment variable %s not found in host environment which is required by %s",
                env_variable,
                self.name,
            )
            raise ValueError(
                f"Environment variable {env_variable} required by {self.name} not found in host environment"
            )
        logger.info("✅ Verified presence of environment variable %s in the host environment", env_variable)

    def verify_tool_installation(self, env):
        return super().verify_tool_installation(env)
