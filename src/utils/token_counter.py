"""Token counting utilities for cost estimation and rate limiting."""

import logging


logger = logging.getLogger(__name__)


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate the number of tokens in a text string.

    Uses a rough heuristic (4 chars per token for English).
    For precise counting, use tiktoken with the specific model's encoding.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4",
) -> float:
    """Estimate the cost in USD for a given token usage.

    These are approximate rates and should be updated as pricing changes.
    """
    pricing = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    }

    rates = pricing.get(model, {"input": 0.01, "output": 0.03})
    cost = (input_tokens / 1000 * rates["input"]) + (
        output_tokens / 1000 * rates["output"]
    )
    return round(cost, 6)
