"""Prompt chaining logic.

Provides :class:`PromptChain` which sequences multiple LLM calls,
feeding each step's output into the next step's template.  Reuses
:class:`BaseLLMClient` and :class:`PromptTemplate` so that chaining
doesn't duplicate generation or templating code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.llm.base import BaseLLMClient
from src.prompt_engineering.templates import PromptTemplate
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChainStep:
    """One step in a prompt chain.

    Args:
        name: Human-readable step name (used in logs and results).
        template: The :class:`PromptTemplate` for this step.
        output_key: Variable name under which this step's result is
            stored so later steps can reference it.
        system_prompt: Optional system prompt for the LLM call.
    """

    name: str
    template: PromptTemplate
    output_key: str = "output"
    system_prompt: Optional[str] = None


class PromptChain:
    """Execute a sequence of LLM calls where each step's output feeds the next.

    Args:
        client: The :class:`BaseLLMClient` to use for generation.
        steps: Ordered list of :class:`ChainStep` objects.
    """

    def __init__(
        self,
        client: BaseLLMClient,
        steps: Optional[list[ChainStep]] = None,
    ) -> None:
        self.client = client
        self.steps: list[ChainStep] = steps or []
        self.results: list[dict[str, Any]] = []

    def add_step(
        self,
        name: str,
        template: PromptTemplate,
        output_key: str = "output",
        system_prompt: Optional[str] = None,
    ) -> None:
        """Append a step to the chain."""
        self.steps.append(
            ChainStep(
                name=name,
                template=template,
                output_key=output_key,
                system_prompt=system_prompt,
            )
        )

    def run(self, initial_vars: Optional[dict[str, str]] = None) -> list[dict[str, Any]]:
        """Execute the chain, returning a list of per-step result dicts.

        Each result dict contains ``step``, ``prompt``, ``response`` (the
        full :meth:`BaseLLMClient.generate` dict), and the accumulated
        ``variables`` snapshot.
        """
        variables: dict[str, str] = dict(initial_vars or {})
        self.results = []

        for step in self.steps:
            prompt = step.template.render(**variables)
            logger.info("Chain step %r: prompt length %d chars", step.name, len(prompt))

            response = self.client.generate(prompt, system_prompt=step.system_prompt)
            variables[step.output_key] = response["text"]

            self.results.append(
                {
                    "step": step.name,
                    "prompt": prompt,
                    "response": response,
                    "variables": dict(variables),
                }
            )
        return self.results

    def get_final_output(self) -> Optional[str]:
        """Return the text output of the last step, or ``None``."""
        if not self.results:
            return None
        return self.results[-1]["response"]["text"]
