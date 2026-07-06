"""Shared token-counting helpers for prompt budgeting and cost estimation."""

from __future__ import annotations

from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Rough per-token costs (USD) by model family — kept as a reference table so
# both LLM clients and prompt-engineering modules can estimate costs without
# duplicating the data.
_COST_PER_1K: dict[str, dict[str, float]] = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
}


def count_tokens(text: str, method: str = "approximate") -> int:
    """Return an estimated token count for *text*.

    Args:
        text: The input string.
        method: ``"approximate"`` uses a simple whitespace/punctuation
            heuristic (no dependency required); ``"tiktoken"`` uses the
            ``tiktoken`` library for an accurate GPT count.
    """
    if method == "tiktoken":
        try:
            import tiktoken  # type: ignore[import-untyped]

            enc = tiktoken.encoding_for_model("gpt-4")
            return len(enc.encode(text))
        except ImportError:
            logger.warning("tiktoken not installed, falling back to approximate count")

    # ~4 chars per token is a widely-used heuristic.
    return max(1, len(text) // 4)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
) -> Optional[float]:
    """Estimate cost in USD for an API call.

    Returns ``None`` when the model is not in the known cost table.
    """
    costs = _COST_PER_1K.get(model)
    if costs is None:
        return None
    return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1000
