"""Abstract base LLM client.

All provider-specific clients inherit from ``BaseLLMClient`` so that
common concerns — configuration loading, rate limiting, caching, token
counting, error handling, and response validation — live in one place.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Optional

import yaml

from src.handlers.error_handler import (
    TokenLimitExceededError,
    retry_with_backoff,
)
from src.utils.cache import ResponseCache
from src.utils.logger import get_logger
from src.utils.rate_limiter import RateLimiter
from src.utils.token_counter import count_tokens, estimate_cost

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = Path("config/model_config.yaml")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load a YAML configuration file.

    Shared by all clients so config-loading logic isn't duplicated.
    """
    if not path.exists():
        logger.warning("Config file %s not found, using defaults", path)
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


class BaseLLMClient(abc.ABC):
    """Abstract base for LLM provider clients.

    Subclasses only need to implement :meth:`_call_api` and
    :meth:`_parse_response`; everything else (rate limiting, caching,
    token counting, retry, validation) is handled here.

    Args:
        model: Model identifier (e.g. ``"gpt-4"`` or ``"claude-3-sonnet"``).
        api_key: Provider API key.  If ``None``, looked up from *config*.
        config_path: Path to the YAML config file.
        max_tokens: Maximum tokens for the response.
        temperature: Sampling temperature.
        rate_limiter: Optional shared :class:`RateLimiter` instance.
        cache: Optional shared :class:`ResponseCache` instance.
        use_cache: Whether to cache responses (default ``True``).
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        config_path: Path = DEFAULT_CONFIG_PATH,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        rate_limiter: Optional[RateLimiter] = None,
        cache: Optional[ResponseCache] = None,
        use_cache: bool = True,
    ) -> None:
        self.config = load_config(config_path)
        self.model = model
        self.api_key = api_key or self._resolve_api_key()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.rate_limiter = rate_limiter or RateLimiter()
        self.cache = cache or ResponseCache() if use_cache else None
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _resolve_api_key(self) -> str:
        """Return the API key from config/environment."""

    @abc.abstractmethod
    def _call_api(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """Send a request to the provider and return the raw response dict."""

    @abc.abstractmethod
    def _parse_response(self, raw: dict[str, Any]) -> str:
        """Extract the assistant's text from a raw API response."""

    # ------------------------------------------------------------------
    # Shared public interface
    # ------------------------------------------------------------------

    @retry_with_backoff()
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a completion, with caching, rate-limiting, and retry.

        Returns a dict with keys ``text``, ``model``, ``input_tokens``,
        ``output_tokens``, and optionally ``cost``.
        """
        input_tokens = count_tokens(prompt)
        if input_tokens > self.max_tokens * 4:
            raise TokenLimitExceededError(
                f"Prompt ({input_tokens} tokens) likely exceeds model context"
            )

        params = self._build_params(**kwargs)

        # Cache lookup
        if self.cache is not None:
            cached = self.cache.get(prompt, self.model, params)
            if cached is not None:
                logger.info("Returning cached response for model=%s", self.model)
                return cached

        # Rate-limit
        self.rate_limiter.wait_if_needed()

        messages = self._build_messages(prompt, system_prompt)
        raw = self._call_api(messages, **params)
        text = self._parse_response(raw)

        output_tokens = count_tokens(text)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        result: dict[str, Any] = {
            "text": text,
            "model": self.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": estimate_cost(input_tokens, output_tokens, self.model),
        }

        if self.cache is not None:
            self.cache.put(prompt, self.model, result, params)

        return result

    def get_usage_stats(self) -> dict[str, Any]:
        """Return cumulative token usage and estimated cost."""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_total_cost": estimate_cost(
                self.total_input_tokens, self.total_output_tokens, self.model
            ),
        }

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_params(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "temperature": kwargs.pop("temperature", self.temperature),
            **kwargs,
        }

    @staticmethod
    def _build_messages(
        prompt: str, system_prompt: Optional[str] = None
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages
