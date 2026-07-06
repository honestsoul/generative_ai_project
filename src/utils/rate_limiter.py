"""Rate limiter for API calls.

Prevents exceeding provider rate limits and protects against
accidental runaway loops that could incur large costs.
"""

import time
import threading
import logging
from collections import deque

from config import get_model_config


logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter for API requests."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 90_000,
    ):
        config = get_model_config().get("rate_limiting", {})
        self.requests_per_minute = requests_per_minute or config.get(
            "requests_per_minute", 60
        )
        self.tokens_per_minute = tokens_per_minute or config.get(
            "tokens_per_minute", 90_000
        )
        self._request_timestamps: deque = deque()
        self._token_usage: deque = deque()
        self._lock = threading.Lock()

    def wait_if_needed(self, estimated_tokens: int = 0) -> None:
        """Block until the request can proceed within rate limits."""
        with self._lock:
            now = time.time()
            window_start = now - 60.0

            # Clean old entries
            while self._request_timestamps and self._request_timestamps[0] < window_start:
                self._request_timestamps.popleft()
            while self._token_usage and self._token_usage[0][0] < window_start:
                self._token_usage.popleft()

            # Check request limit
            if len(self._request_timestamps) >= self.requests_per_minute:
                sleep_time = self._request_timestamps[0] - window_start
                logger.info(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(max(0, sleep_time))

            # Check token limit
            current_tokens = sum(t[1] for t in self._token_usage)
            if current_tokens + estimated_tokens > self.tokens_per_minute:
                sleep_time = self._token_usage[0][0] - window_start
                logger.info(f"Token limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(max(0, sleep_time))

            self._request_timestamps.append(time.time())
            if estimated_tokens > 0:
                self._token_usage.append((time.time(), estimated_tokens))

    def record_usage(self, tokens_used: int) -> None:
        """Record actual token usage after a request completes."""
        with self._lock:
            self._token_usage.append((time.time(), tokens_used))
