"""Shared error handling and retry logic for API calls.

Provides a single ``retry_with_backoff`` decorator and a hierarchy of
domain-specific exceptions so that all LLM clients share the same retry
strategy, back-off timing, and error taxonomy.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

from src.utils.logger import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Base exception for all LLM-related errors."""


class APIConnectionError(LLMError):
    """Network-level failure when calling the provider."""


class APIRateLimitError(LLMError):
    """The provider returned a rate-limit (HTTP 429) response."""


class APIResponseError(LLMError):
    """The provider returned an unexpected or malformed response."""


class TokenLimitExceededError(LLMError):
    """The prompt exceeds the model's maximum context length."""


class AuthenticationError(LLMError):
    """Invalid or missing API key."""


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

# Exceptions that are safe to retry automatically.
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    APIConnectionError,
    APIRateLimitError,
)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable: tuple[type[BaseException], ...] = RETRYABLE_EXCEPTIONS,
) -> Callable[[F], F]:
    """Decorator that retries a function with exponential back-off.

    Args:
        max_retries: Total retry attempts before giving up.
        base_delay: Initial delay in seconds (doubled each retry).
        max_delay: Cap on the delay between retries.
        retryable: Exception types that trigger a retry.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = base_delay
            last_exc: BaseException | None = None
            for attempt in range(1, max_retries + 2):  # +1 for initial try
                try:
                    return func(*args, **kwargs)
                except retryable as exc:
                    last_exc = exc
                    if attempt > max_retries:
                        break
                    logger.warning(
                        "%s failed (attempt %d/%d): %s – retrying in %.1fs",
                        func.__qualname__,
                        attempt,
                        max_retries + 1,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator
