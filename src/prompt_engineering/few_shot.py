"""Few-shot prompt utilities with input validation."""

import logging
from typing import Optional


logger = logging.getLogger(__name__)


class FewShotExample:
    """A single input-output example for few-shot prompting."""

    def __init__(self, input_text: str, output_text: str):
        if not input_text or not input_text.strip():
            raise ValueError("Example input cannot be empty.")
        if not output_text or not output_text.strip():
            raise ValueError("Example output cannot be empty.")
        self.input_text = input_text.strip()
        self.output_text = output_text.strip()

    def format(self, input_prefix: str = "Input", output_prefix: str = "Output") -> str:
        return f"{input_prefix}: {self.input_text}\n{output_prefix}: {self.output_text}"


class FewShotPrompt:
    """Constructs few-shot prompts with validation."""

    MAX_EXAMPLES = 20

    def __init__(
        self,
        instruction: str,
        examples: Optional[list[FewShotExample]] = None,
    ):
        if not instruction or not instruction.strip():
            raise ValueError("Instruction cannot be empty.")
        self.instruction = instruction.strip()
        self.examples: list[FewShotExample] = examples or []

    def add_example(self, input_text: str, output_text: str) -> None:
        """Add a few-shot example with validation."""
        if len(self.examples) >= self.MAX_EXAMPLES:
            raise ValueError(
                f"Cannot add more than {self.MAX_EXAMPLES} examples."
            )
        self.examples.append(FewShotExample(input_text, output_text))

    def render(self, query: str) -> str:
        """Render the complete few-shot prompt for a given query."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        parts = [self.instruction, ""]
        for example in self.examples:
            parts.append(example.format())
            parts.append("")

        parts.append(f"Input: {query.strip()}")
        parts.append("Output:")

        return "\n".join(parts)
