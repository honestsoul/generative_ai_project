"""Few-shot prompt utilities.

Builds few-shot prompts by combining example pairs with a query,
reusing :class:`PromptTemplate` for the overall structure and
:func:`count_tokens` for budget management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.utils.logger import get_logger
from src.utils.token_counter import count_tokens

logger = get_logger(__name__)


@dataclass
class Example:
    """A single input/output demonstration pair."""

    input: str
    output: str
    label: str = ""


@dataclass
class FewShotPrompt:
    """Assembles a few-shot prompt from examples and a query.

    Args:
        instruction: Top-level instruction preceding the examples.
        examples: Ordered list of demonstration pairs.
        query_prefix: Text placed before the actual user query.
        max_tokens: Token budget; examples are pruned to fit.
    """

    instruction: str = ""
    examples: list[Example] = field(default_factory=list)
    query_prefix: str = "Input:"
    max_tokens: Optional[int] = None

    def add_example(self, input_text: str, output_text: str, label: str = "") -> None:
        """Append a new example."""
        self.examples.append(Example(input=input_text, output=output_text, label=label))

    def _format_example(self, ex: Example) -> str:
        header = f"[{ex.label}] " if ex.label else ""
        return f"{header}Input: {ex.input}\nOutput: {ex.output}"

    def _select_examples(self, query: str) -> list[Example]:
        """Return as many examples as fit within the token budget."""
        if self.max_tokens is None:
            return list(self.examples)

        base_tokens = count_tokens(self.instruction + self.query_prefix + query)
        budget = self.max_tokens - base_tokens
        selected: list[Example] = []
        for ex in self.examples:
            ex_tokens = count_tokens(self._format_example(ex))
            if budget - ex_tokens < 0:
                break
            selected.append(ex)
            budget -= ex_tokens
        if len(selected) < len(self.examples):
            logger.info(
                "Pruned examples from %d to %d to fit token budget",
                len(self.examples),
                len(selected),
            )
        return selected

    def render(self, query: str) -> str:
        """Build the full few-shot prompt string."""
        parts: list[str] = []
        if self.instruction:
            parts.append(self.instruction)
        for ex in self._select_examples(query):
            parts.append(self._format_example(ex))
        parts.append(f"{self.query_prefix} {query}")
        parts.append("Output:")
        return "\n\n".join(parts)
