"""Shared rate limiter for API calls."""

import time
from threading import Lock

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Token-bucket rate limiter shared across LLM clients.

    Args:
        max_calls: Maximum number of calls allowed per ``period`` seconds.
        period: Time window in seconds.
    """

    def __init__(self, max_calls: int = 60, period: float = 60.0) -> None:
        self.max_calls = max_calls
        self.period = period
        self._calls: list[float] = []
        self._lock = Lock()

    def _purge_old_calls(self) -> None:
        now = time.monotonic()
        self._calls = [t for t in self._calls if now - t < self.period]

    def wait_if_needed(self) -> None:
        """Block until a call is allowed under the rate limit."""
        with self._lock:
            self._purge_old_calls()
            if len(self._calls) >= self.max_calls:
                sleep_time = self.period - (time.monotonic() - self._calls[0])
                if sleep_time > 0:
                    logger.info("Rate limit reached, sleeping %.2fs", sleep_time)
                    time.sleep(sleep_time)
                self._purge_old_calls()
            self._calls.append(time.monotonic())

    @property
    def remaining_calls(self) -> int:
        """Number of calls still available in the current window."""
        with self._lock:
            self._purge_old_calls()
            return max(0, self.max_calls - len(self._calls))
