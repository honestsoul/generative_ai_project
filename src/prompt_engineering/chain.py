"""Prompt chaining logic with validation between steps."""

import logging
from typing import Callable, Optional


logger = logging.getLogger(__name__)


class ChainStep:
    """A single step in a prompt chain."""

    def __init__(
        self,
        name: str,
        prompt_template: str,
        output_key: str,
        validator: Optional[Callable[[str], bool]] = None,
    ):
        if not name or not name.strip():
            raise ValueError("Step name cannot be empty.")
        if not prompt_template or not prompt_template.strip():
            raise ValueError("Prompt template cannot be empty.")
        if not output_key or not output_key.strip():
            raise ValueError("Output key cannot be empty.")

        self.name = name
        self.prompt_template = prompt_template
        self.output_key = output_key
        self.validator = validator

    def format_prompt(self, context: dict) -> str:
        """Format the prompt template with values from context."""
        prompt = self.prompt_template
        for key, value in context.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))
        return prompt

    def validate_output(self, output: str) -> bool:
        """Validate step output if a validator is provided."""
        if self.validator is None:
            return True
        return self.validator(output)


class PromptChain:
    """Chains multiple prompts together, passing outputs as context."""

    MAX_STEPS = 10

    def __init__(self, steps: Optional[list[ChainStep]] = None):
        self.steps: list[ChainStep] = steps or []

    def add_step(self, step: ChainStep) -> None:
        """Add a step to the chain."""
        if len(self.steps) >= self.MAX_STEPS:
            raise ValueError(f"Cannot add more than {self.MAX_STEPS} steps.")
        self.steps.append(step)

    def get_prompts(self, initial_context: dict) -> list[tuple[str, str]]:
        """Generate all prompts in the chain given initial context.

        Returns a list of (step_name, formatted_prompt) tuples.
        Note: In practice, each step's output must be obtained from the LLM
        before the next step can be formatted. This method is for preview only
        when all context values are already known.
        """
        context = dict(initial_context)
        prompts = []
        for step in self.steps:
            prompt = step.format_prompt(context)
            prompts.append((step.name, prompt))
        return prompts
