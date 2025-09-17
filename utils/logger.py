from enum import StrEnum


# Create a EMOJI enums for differnt types logger
class LogLevelEmoji(StrEnum):
    INFO = "ℹ️"
    WARNING = "⚠️"
    ERROR = "❌"
    CRITICAL = "🚨"
    DEBUG = "🐛"
    COMPLETED = "✅"
