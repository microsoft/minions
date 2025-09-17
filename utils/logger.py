import logging
from logging import Formatter, Logger

# Step 1 & 2: Define and register the new custom logging level
COMPLETED_LEVEL_NUM = 25
logging.addLevelName(COMPLETED_LEVEL_NUM, "COMPLETED")


# Step 3: Add a custom method to the Logger class
def completed(self, message, *args, **kwargs):
    if self.isEnabledFor(COMPLETED_LEVEL_NUM):
        self._log(COMPLETED_LEVEL_NUM, message, args, **kwargs)


Logger.completed = completed


class EmojiFormatter(Formatter):
    """A custom formatter to add emojis based on log level."""

    def format(self, record):
        if record.levelno == logging.INFO:
            record.levelname = "ℹ️  INFO"
        elif record.levelno == logging.WARNING:
            record.levelname = "⚠️  WARNING"
        elif record.levelno == logging.ERROR:
            record.levelname = "❌  ERROR"
        elif record.levelno == logging.CRITICAL:
            record.levelname = "🚨  CRITICAL"
        elif record.levelno == logging.DEBUG:
            record.levelname = "🐛  DEBUG"
        # Add the new COMPLETED level and its emoji
        elif record.levelno == COMPLETED_LEVEL_NUM:
            record.levelname = "✅ COMPLETED"
        return super().format(record)


# Step 1: Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Step 2: Create a handler (e.g., for the console)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Step 3: Create an instance of our custom formatter
emoji_formatter = EmojiFormatter(
    fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# Step 4: Add the formatter to the handler
console_handler.setFormatter(emoji_formatter)

# Step 5: Add the handler to the logger
logger.addHandler(console_handler)
