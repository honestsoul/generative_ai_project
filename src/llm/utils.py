"""Shared LLM utility functions.

Helpers used by multiple LLM clients or by callers that don't need a
full client instance (e.g. quick model selection, message formatting).
"""

from __future__ import annotations

from typing import Any, Optional

from src.llm.base import BaseLLMClient, load_config
from src.utils.logger import get_logger
from src.utils.token_counter import count_tokens

logger = get_logger(__name__)

# Registry populated lazily so we don't import heavy client modules at
# module level unless needed.
_CLIENT_REGISTRY: dict[str, type[BaseLLMClient]] = {}


def register_client(provider: str, cls: type[BaseLLMClient]) -> None:
    """Register a client class under a provider name."""
    _CLIENT_REGISTRY[provider] = cls


def _ensure_registry() -> None:
    if _CLIENT_REGISTRY:
        return
    from src.llm.claude_client import ClaudeClient
    from src.llm.openai_client import OpenAIClient

    register_client("anthropic", ClaudeClient)
    register_client("openai", OpenAIClient)


def create_client(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> BaseLLMClient:
    """Factory that returns the right client for *provider*.

    >>> client = create_client("openai", "gpt-4", api_key="sk-...")
    """
    _ensure_registry()
    cls = _CLIENT_REGISTRY.get(provider)
    if cls is None:
        raise ValueError(
            f"Unknown provider {provider!r}. "
            f"Available: {sorted(_CLIENT_REGISTRY)}"
        )
    return cls(model=model, api_key=api_key, **kwargs)


def format_chat_messages(
    conversation: list[dict[str, str]],
    system_prompt: Optional[str] = None,
) -> list[dict[str, str]]:
    """Build a normalized message list from a conversation history.

    Ensures a system message appears first (if provided) and that roles
    alternate correctly.
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(conversation)
    return messages


def truncate_to_token_limit(
    text: str, max_tokens: int, method: str = "approximate"
) -> str:
    """Truncate *text* so its token count stays within *max_tokens*."""
    tokens = count_tokens(text, method=method)
    if tokens <= max_tokens:
        return text
    ratio = max_tokens / tokens
    cut = int(len(text) * ratio)
    truncated = text[:cut]
    logger.info("Truncated text from ~%d to ~%d tokens", tokens, max_tokens)
    return truncated


def select_model(task: str = "general") -> dict[str, str]:
    """Suggest a provider/model pair based on *task* type.

    Uses heuristics; callers can override. Returned dict has keys
    ``provider`` and ``model``.
    """
    config = load_config()
    defaults = config.get("defaults", {})

    task_map: dict[str, dict[str, str]] = {
        "general": {"provider": "openai", "model": "gpt-3.5-turbo"},
        "complex": {"provider": "openai", "model": "gpt-4"},
        "creative": {"provider": "anthropic", "model": "claude-3-sonnet"},
        "fast": {"provider": "anthropic", "model": "claude-3-haiku"},
    }
    suggestion = task_map.get(task, task_map["general"])
    return {
        "provider": defaults.get("provider", suggestion["provider"]),
        "model": defaults.get("model", suggestion["model"]),
    }
