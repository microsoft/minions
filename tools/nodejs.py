from tools import Tools


# Example of a concrete class inheriting from Tools
class NodeJS(Tools):
    def __init__(self, installation_command, verification_command, usage_instructions):
        super().__init__(installation_command, verification_command, usage_instructions)

