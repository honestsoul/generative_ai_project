"""OpenAI GPT client implementation with secure API key handling."""

import logging
from typing import Optional

from openai import OpenAI

from src.llm.base import BaseLLMClient


logger = logging.getLogger(__name__)


class GPTClient(BaseLLMClient):
    """OpenAI GPT client.

    API key is loaded from the OPENAI_API_KEY environment variable.
    Never pass API keys as constructor arguments or hardcode them.
    """

    PROVIDER = "openai"

    def __init__(self, model: Optional[str] = None):
        super().__init__(model=model)
        self._client = OpenAI(api_key=self._api_key)

    def complete(self, prompt: str, **kwargs) -> str:
        """Generate a completion using OpenAI's API."""
        validated_prompt = self._validate_input(prompt)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": validated_prompt}],
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )
        return response.choices[0].message.content

    def chat(self, messages: list[dict], **kwargs) -> str:
        """Generate a chat response using OpenAI's API."""
        if not messages:
            raise ValueError("Messages list cannot be empty.")

        for msg in messages:
            if "role" not in msg or "content" not in msg:
                raise ValueError(
                    "Each message must have 'role' and 'content' keys."
                )
            self._validate_input(msg["content"])

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )
        return response.choices[0].message.content
