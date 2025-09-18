# Bot
    Bot will accept "name", "permission" and "system_prompt", "llm", "environment", "additional_tools", "folder_to_mount
        "Bot_type" => "READING_Bot" | "WRITING_Bot" | "BROWSING_Bot" | "CUSTOM_Bot"
        "permission" => "READ_ONLY" | "READ_WRITE" (optional field)
        "model_name" => { // Based on provider the Bot will instantiate llm instance
            provider: "openai" 
            model: "gpt-4" | "gpt-3.5-turbo" | "claude",
        }
        <!-- LLM should not give any interactive commands like vim -->
        "system_prompt" => string (Optional argument)
            when not passed will create default system prompt based on Bot_type and permission
        "environment" => environment instance (Optional argument)
            when not passed will create docker environment with default arguments as per PERMISSION based on Bot_type
            and tools will be installed one by one after environment creation
        "additional_tools" => array of tool definitions (Optional argument)
            when passed will create docker environment with default arguments as per PERMISSION and with default "tool" installations with additional tools installed

    Bot will expose "run" method which will accept a "task", "timeout" and "max_iterations"
        will call llm task as user_prompt and get response
        get the response and call environment execute method with the command
        get the output and call llm with previous_communications and get the response
        repeat the above steps until llm response is final answer or max_iterations reached or timeout reached
        ensure to log every communication between llm and environment

        <!-- In the end llm should provide status "DONE" status in one object property with results in another property -->

        will return { status: bool, result: string, error: string | None, log: log of communication between llm and environment }
        


# environment
    environment will accept "folder_to_mount", "permission"
        "folder_to_mount" => string (path to the folder to mount inside docker)
        "permission" => "READ_ONLY" | "READ_WRITE"

    environment will expose "execute" method which will accept command, timeout and return {
        return_code: int,
        output: string,
        error: string | None
    }
        will execute the command inside the docker with a default timeout and return the output
    environment will expose "kill" method which will kill the environment
        will stop the docker container and remove it

# llm class

    llm will accept "system_prompt" and "deployment_name"
    system_prompt => string, 
    deployment_name => string (like my-custom-gpt5-Bot)

    llm will expose ask method which will accept "message" and will return llm response in particular json format
        the method will append each message to messages array and each response also will be appended to the messages array, 
    llm will expose clear history method which will clear the messages array except the system prompt
    

# tool definition singleton class
    tool definition will accept "installation_command", "verification_command", "usage_instructions"

Whenever I am writing code 
```py
import logging
logger = logging.getLogger(__name__)
logger.info("This is an info message")
```