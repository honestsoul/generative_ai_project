"""Error handling package.

Re-exports the exception hierarchy and retry decorator.
"""

from src.handlers.error_handler import (
    APIConnectionError,
    APIRateLimitError,
    APIResponseError,
    AuthenticationError,
    LLMError,
    TokenLimitExceededError,
    retry_with_backoff,
)

__all__ = [
    "APIConnectionError",
    "APIRateLimitError",
    "APIResponseError",
    "AuthenticationError",
    "LLMError",
    "TokenLimitExceededError",
    "retry_with_backoff",
]
