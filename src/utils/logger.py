"""Logging utilities with security-aware formatting.

Ensures that sensitive data (API keys, tokens) is never written to logs.
"""

import logging
import re
import sys

from config import get_logging_config


SENSITIVE_PATTERNS = [
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "sk-***"),
    (re.compile(r"sk-ant-[a-zA-Z0-9\-]{20,}"), "sk-ant-***"),
    (re.compile(r"(?i)(bearer\s+)\S+"), r"\1***"),
    (re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*\S+"),
     r"\1=***"),
]


class SanitizingFormatter(logging.Formatter):
    """Log formatter that redacts sensitive data from log messages."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        for pattern, replacement in SENSITIVE_PATTERNS:
            message = pattern.sub(replacement, message)
        return message


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure application logging with security-aware formatting."""
    config = get_logging_config()
    log_level = getattr(logging, config.get("level", level).upper(), logging.INFO)

    formatter = SanitizingFormatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(handler)

    return root_logger
