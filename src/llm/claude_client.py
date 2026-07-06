"""Anthropic Claude LLM client.

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

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"


class ClaudeClient(BaseLLMClient):
    """Client for Anthropic's Claude models."""

    def _resolve_api_key(self) -> str:
        key = self.config.get("anthropic", {}).get("api_key") or os.getenv(
            "ANTHROPIC_API_KEY", ""
        )
        if not key:
            raise AuthenticationError("Anthropic API key not configured")
        return key

    def _call_api(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> dict[str, Any]:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        user_messages = [m for m in messages if m["role"] != "system"]

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }
        body: dict[str, Any] = {
            "model": self.model,
            "messages": user_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if system_parts:
            body["system"] = system_parts[0]

        try:
            resp = requests.post(
                ANTHROPIC_API_URL, headers=headers, json=body, timeout=120
            )
        except requests.ConnectionError as exc:
            raise APIConnectionError(str(exc)) from exc

        if resp.status_code == 429:
            raise APIRateLimitError("Anthropic rate limit hit")
        if resp.status_code == 401:
            raise AuthenticationError("Invalid Anthropic API key")
        if resp.status_code != 200:
            raise APIResponseError(
                f"Anthropic API error {resp.status_code}: {resp.text}"
            )
        return resp.json()

    def _parse_response(self, raw: dict[str, Any]) -> str:
        try:
            return raw["content"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise APIResponseError(f"Unexpected response format: {raw}") from exc
