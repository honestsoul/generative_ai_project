"""Shared utilities for LLM clients.

Security note: This module must never log or expose API keys.
"""

import re
import logging


logger = logging.getLogger(__name__)


def sanitize_prompt(prompt: str) -> str:
    """Remove potentially dangerous content from prompts.

    Strips common injection patterns while preserving legitimate content.
    """
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string.")
    return prompt.strip()


def mask_sensitive_data(text: str) -> str:
    """Mask patterns that look like API keys or secrets in log output."""
    patterns = [
        (r"sk-[a-zA-Z0-9]{20,}", "sk-***REDACTED***"),
        (r"sk-ant-[a-zA-Z0-9\-]{20,}", "sk-ant-***REDACTED***"),
        (r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*\S+",
         r"\1=***REDACTED***"),
    ]
    masked = text
    for pattern, replacement in patterns:
        masked = re.sub(pattern, replacement, masked)
    return masked


def validate_model_name(model: str, provider: str) -> bool:
    """Validate that a model name follows expected patterns."""
    allowed_patterns = {
        "openai": r"^(gpt-4|gpt-3\.5|o1|o3)[\w\-]*$",
        "anthropic": r"^claude-[\w\-]+$",
    }
    pattern = allowed_patterns.get(provider)
    if not pattern:
        return False
    return bool(re.match(pattern, model))
