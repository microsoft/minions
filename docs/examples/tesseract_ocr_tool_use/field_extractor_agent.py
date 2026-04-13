from microbots import WritingBot
import os
import logging
from microbots.tools.tool_yaml_parser import parse_tool_definition
from microbots.tools.internal_tool import Tool
from microbots.tools.tool import EnvFileCopies


def setup_logging():
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)

    info_handler = logging.FileHandler(os.path.join(log_dir, "info.log"))
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(log_format)

    debug_handler = logging.FileHandler(os.path.join(log_dir, "debug.log"))
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(log_format)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(info_handler)
    root_logger.addHandler(debug_handler)


def main():
    setup_logging()
    tesseract_ocr_yaml = os.path.join(os.getcwd(), "tesseract_ocr.yaml")

    tool = parse_tool_definition(tesseract_ocr_yaml)
    
    bot = WritingBot(
        model = "azure-openai/mini-swe-agent-gpt5",
        folder_to_mount = os.path.join(os.getcwd(), "pngs"),
        additional_tools=[tool]
    )
    bot.run(
        task="Extract input fields from the pdf_page-01.png file and generate a JSON output with field name, field value, and confidence score. The fields to extract are: Name, expected data type, expected length. I'll use this information to build an UI form for user to enter the data.",
        timeout_in_seconds=300,
        max_iterations=50
    )


if __name__ == "__main__":
    main()