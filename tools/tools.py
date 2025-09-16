from abc import ABC


class Tools(ABC):

    def __init__(self, installation_command, verification_command, usage_instructions):
        self.installation_command = installation_command
        self.verification_command = verification_command
        self.usage_instructions = usage_instructions
