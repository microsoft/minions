import logging

logger = logging.getLogger(" 🔧 Compact Tool ")

from microbots.MicroBot import MicroBot  # noqa: E402
from microbots.external_tools.external_tool import ExternalTool  # noqa: E402


TOOL_USAGE_INSTRUCTIONS = """
Use this tool to summarize conversations into notes. It will update the notes section of the system prompt surrounded by <notes></notes> tags.

Usage:
    compact <number_of_recent_messages_to_preserve> "<note_content>"

    <number_of_recent_messages_to_preserve>: The number of recent user-assistant pairs to keep in the conversation history.
    <note_content>: The content of the note to be created.

Example:
    compact 5 "Meeting notes from today's discussion on project milestones."
It will preserve the last 5 user-assistant message pairs in the conversation history. And update the notes with the provided content.

    compact 0 "Searched file tool.py is not present in the codebase."
It will clear the entire conversation history and create a note with the provided content.

Notes:
- Ensure that the <note_content> is enclosed in double quotes.
- This tool replaces existing notes with the new content provided. So, your notes should be combination of previous notes and new information explored recently.
- Use this often to keep your notes updated with the latest information from the conversation. It's wise to maintain less conversation history for better performance.
"""

class Compact(ExternalTool):
    name: str = "Compact"
    command: str = "compact"
    description: str = (
        "A tool for generating notes from conversations."
        "It allows LLM to summarize conversations into notes."
    )
    usage_instructions_to_llm: str = TOOL_USAGE_INSTRUCTIONS

    def call(self, microbot: MicroBot, command: str) -> str:
        logger.debug("Compact tool called with command: %s", command)
        try:
            parts = command.split(" ", 2)
            if len(parts) != 3:
                logger.error("Invalid command format: %s", command)
                return "Error: Invalid command format. Use: compact <number_of_recent_messages_to_preserve> \"<note_content>\""

            _, num_messages_str, note_content = parts
            logger.debug("Parsed num_messages: %s, note_content: %s", num_messages_str, note_content)
            num_messages = int(num_messages_str)

            if num_messages < 0:
                logger.error("Negative number of messages to preserve: %d", num_messages)
                return "Error: <number_of_recent_messages_to_preserve> must be a non-negative integer."

            # Update notes in the system prompt
            system_prompt = microbot.llm_interface.messages[0]["content"]
            logger.debug("Current system prompt before updating notes: %s", system_prompt)

            start_tag = "<notes>"
            end_tag = "</notes>"
            logger.debug("Searching for notes section between %s and %s", start_tag, end_tag)

            start_index = system_prompt.find(start_tag)
            end_index = system_prompt.find(end_tag)

            if start_index == -1 or end_index == -1:
                logger.error("Notes section not found in the system prompt.")
                return "Error: Notes section not found in the system prompt."

            new_system_prompt = (
                system_prompt[: start_index + len(start_tag)]
                + "\n"
                + note_content
                + "\n"
                + system_prompt[end_index:]
            )
            logger.debug("Updated system prompt with new notes: %s", new_system_prompt)
            microbot.llm_interface.messages[0]["content"] = new_system_prompt

            # Compact conversation history
            history = microbot.llm_interface.messages[1:]  # Exclude system prompt
            user_assistant_pairs = [
                (history[i], history[i + 1]) for i in range(0, len(history), 2)
            ]
            # If num_messages is greater than available pairs, python slicing will handle it gracefully
            preserved_pairs = user_assistant_pairs[-num_messages:]
            logger.debug("Preserved user-assistant pairs: %s", preserved_pairs)

            new_history = []
            for user_msg, assistant_msg in preserved_pairs:
                new_history.append(user_msg)
                new_history.append(assistant_msg)

            microbot.llm_interface.messages = list(
                [microbot.llm_interface.messages[0]] + new_history
            )

            microbot.llm_interface.messages[-1] = {
                "role": "assistant",
                "content": "compact command call (placeholder text)",
            }
            logger.debug("Conversation history after compaction: %s", microbot.llm_interface.messages)

            logger.info("Notes updated and conversation history compacted successfully.")
            return f"Notes updated and conversation history compacted to preserve the last {num_messages} user-assistant pairs."

        except ValueError:
            logger.error("Invalid number format for <number_of_recent_messages_to_preserve>: %s", num_messages_str)
            return "Error: <number_of_recent_messages_to_preserve> must be an integer."
        except Exception as e:
            logger.exception("An unexpected error occurred: %s", str(e))
            return f"An unexpected error occurred: {str(e)}"