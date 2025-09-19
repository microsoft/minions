from microbot.tool_definitions.base_tool import BaseTool


class Node(BaseTool):

    _instance = None  # Class-level attribute for Singleton
    # These properties must be defined to satisfy the abstract base class
    name = "node"
    installation_command = "sudo apt update && sudo apt install nodejs npm"
    verification_command = "node -v"
    usage_instructions_to_llm = "To use Node.js, you can run 'node <your_script.js>'"

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Node, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # We can still use __init__ for any additional setup if needed
        if not hasattr(self, "initialized"):
            self.initialized = True
            print(f"Singleton instance initialized with name: {self.name}")
        else:
            print(f"Attempted to re-initialize an existing instance.")
