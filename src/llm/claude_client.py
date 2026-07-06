"""Anthropic Claude client implementation with secure API key handling."""

import logging
from typing import Optional

import anthropic

from src.llm.base import BaseLLMClient


logger = logging.getLogger(__name__)


class ClaudeClient(BaseLLMClient):
    """Anthropic Claude client.

    API key is loaded from the ANTHROPIC_API_KEY environment variable.
    Never pass API keys as constructor arguments or hardcode them.
    """

    PROVIDER = "anthropic"

    def __init__(self, model: Optional[str] = None):
        super().__init__(model=model)
        self._client = anthropic.Anthropic(api_key=self._api_key)

    def complete(self, prompt: str, **kwargs) -> str:
        """Generate a completion using Anthropic's API."""
        validated_prompt = self._validate_input(prompt)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            messages=[{"role": "user", "content": validated_prompt}],
        )
        return response.content[0].text

    def chat(self, messages: list[dict], **kwargs) -> str:
        """Generate a chat response using Anthropic's API."""
        if not messages:
            raise ValueError("Messages list cannot be empty.")

        for msg in messages:
            if "role" not in msg or "content" not in msg:
                raise ValueError(
                    "Each message must have 'role' and 'content' keys."
                )
            self._validate_input(msg["content"])

        response = self._client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            messages=messages,
        )
        return response.content[0].text
