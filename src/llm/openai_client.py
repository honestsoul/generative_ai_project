"""OpenAI GPT LLM client.

Thin subclass of :class:`BaseLLMClient` — only provider-specific API
call mechanics and response parsing live here; everything else (rate
limiting, caching, retry, token counting) is inherited.
"""

from __future__ import annotations

import os
from typing import Any

import requests

from src.handlers.error_handler import (
    APIConnectionError,
    APIRateLimitError,
    APIResponseError,
    AuthenticationError,
)
from src.llm.base import BaseLLMClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI's GPT models."""

    def _resolve_api_key(self) -> str:
        key = self.config.get("openai", {}).get("api_key") or os.getenv(
            "OPENAI_API_KEY", ""
        )
        if not key:
            raise AuthenticationError("OpenAI API key not configured")
        return key

    def _call_api(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        try:
            resp = requests.post(
                OPENAI_API_URL, headers=headers, json=body, timeout=120
            )
        except requests.ConnectionError as exc:
            raise APIConnectionError(str(exc)) from exc

        if resp.status_code == 429:
            raise APIRateLimitError("OpenAI rate limit hit")
        if resp.status_code == 401:
            raise AuthenticationError("Invalid OpenAI API key")
        if resp.status_code != 200:
            raise APIResponseError(
                f"OpenAI API error {resp.status_code}: {resp.text}"
            )
        return resp.json()

    def _parse_response(self, raw: dict[str, Any]) -> str:
        try:
            return raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise APIResponseError(f"Unexpected response format: {raw}") from exc
