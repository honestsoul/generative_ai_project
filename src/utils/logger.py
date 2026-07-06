"""Shared logging utilities for the generative AI project."""

import logging
import sys
from pathlib import Path
from typing import Optional


_LOGGERS: dict[str, logging.Logger] = {}

DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LEVEL = logging.INFO


def get_logger(
    name: str,
    level: int = DEFAULT_LEVEL,
    log_file: Optional[str] = None,
    fmt: str = DEFAULT_FORMAT,
) -> logging.Logger:
    """Return a configured logger, reusing existing ones by name.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).
        level: Logging level.
        log_file: Optional path to a file for log output.
        fmt: Log message format string.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(fmt)

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        logger.addHandler(console)

        if log_file is not None:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger
