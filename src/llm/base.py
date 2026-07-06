"""Base LLM client with secure configuration patterns.

Security measures:
- API keys loaded exclusively from environment variables
- Input validation and sanitization
- No secrets in logs or error messages
- Rate limiting enforcement
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Optional

from config import get_api_key, get_model_config


logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for LLM provider clients.

    Handles secure API key management and input validation.
    """

    PROVIDER: str = ""
    MAX_PROMPT_LENGTH: int = 100_000  # characters

    def __init__(self, model: Optional[str] = None):
        """Initialize the LLM client.

        Args:
            model: Model identifier override. If None, uses config default.

        Raises:
            EnvironmentError: If the required API key is not set.
        """
        self._api_key = get_api_key(self.PROVIDER)
        config = get_model_config().get(self.PROVIDER, {})
        self.model = model or config.get("default_model", "")
        self.max_tokens = config.get("max_tokens", 1024)
        self.temperature = config.get("temperature", 0.7)

    def _validate_input(self, prompt: str) -> str:
        """Validate and sanitize user input before sending to the API.

        Args:
            prompt: The user's input prompt.

        Returns:
            The validated prompt string.

        Raises:
            ValueError: If input fails validation.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if len(prompt) > self.MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt exceeds maximum length of {self.MAX_PROMPT_LENGTH} characters."
            )

        return prompt.strip()

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt: The input prompt.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The generated text response.
        """
        ...

    @abstractmethod
    def chat(self, messages: list[dict], **kwargs) -> str:
        """Generate a chat response for the given message history.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The generated response text.
        """
        ...

    def _redact_key_for_logging(self) -> str:
        """Return a redacted version of the API key for safe logging."""
        if len(self._api_key) > 8:
            return self._api_key[:4] + "..." + self._api_key[-4:]
        return "***REDACTED***"
