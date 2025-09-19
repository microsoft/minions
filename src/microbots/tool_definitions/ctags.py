from microbots.tool_definitions.base_tool import BaseTool


class Ctags(BaseTool):

    _instance = None  # Class-level attribute to store the single instance
    name = "ctags"
    installation_command = "sudo apt install universal-ctags"
    verification_command = "ctags --version"
    usage_instructions_to_llm = (
        "To use ctags, you can run 'ctags -R .' in your project directory"
    )

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Ctags, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # We can still use __init__ for any additional setup if needed
        if not hasattr(self, "initialized"):
            self.initialized = True
            print(f"Singleton instance initialized with name: {self.name}")
        else:
            print(f"Attempted to re-initialize an existing instance.")
