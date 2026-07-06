"""Error handling for LLM API calls.

Security measures:
- Never expose API keys or internal details in error messages
- Sanitize error messages before logging
- Provide safe, user-friendly error responses
"""

import logging
import functools
from typing import Callable

from src.llm.utils import mask_sensitive_data


logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for LLM-related errors."""

    def __init__(self, message: str, provider: str = "unknown"):
        self.provider = provider
        # Sanitize the message to prevent leaking secrets
        safe_message = mask_sensitive_data(message)
        super().__init__(safe_message)


class RateLimitError(LLMError):
    """Raised when API rate limits are exceeded."""
    pass


class AuthenticationError(LLMError):
    """Raised when API authentication fails.

    Never includes the actual key in the error message.
    """
    pass


class InvalidInputError(LLMError):
    """Raised when user input fails validation."""
    pass


def safe_api_call(func: Callable) -> Callable:
    """Decorator that catches API exceptions and re-raises safe versions.

    Ensures no sensitive information leaks through exception messages.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (LLMError, ValueError):
            raise
        except Exception as e:
            error_msg = str(e)
            safe_msg = mask_sensitive_data(error_msg)
            logger.error(f"API call failed: {safe_msg}")
            raise LLMError(
                f"API call failed: {safe_msg}",
                provider="unknown",
            ) from None

    return wrapper
