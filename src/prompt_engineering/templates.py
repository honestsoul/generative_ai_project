"""Prompt template management with input validation."""

import re
import logging
from typing import Optional

from config import get_prompt_templates


logger = logging.getLogger(__name__)


class PromptTemplate:
    """A prompt template with variable substitution and validation."""

    def __init__(self, template: str, name: Optional[str] = None):
        if not template or not template.strip():
            raise ValueError("Template cannot be empty.")
        self.template = template
        self.name = name or "unnamed"
        self._variables = set(re.findall(r"\{(\w+)\}", template))

    @property
    def variables(self) -> set[str]:
        """Return the set of variable names in this template."""
        return self._variables.copy()

    def render(self, **kwargs) -> str:
        """Render the template with the given variables.

        Args:
            **kwargs: Variable name-value pairs.

        Returns:
            The rendered prompt string.

        Raises:
            ValueError: If required variables are missing.
        """
        missing = self._variables - set(kwargs.keys())
        if missing:
            raise ValueError(
                f"Missing template variables: {', '.join(sorted(missing))}"
            )

        rendered = self.template
        for key, value in kwargs.items():
            if key in self._variables:
                if not isinstance(value, str):
                    value = str(value)
                rendered = rendered.replace(f"{{{key}}}", value)

        return rendered


def load_templates() -> dict[str, PromptTemplate]:
    """Load all prompt templates from configuration."""
    config = get_prompt_templates()
    templates = {}
    for name, template_str in config.items():
        if isinstance(template_str, str):
            templates[name] = PromptTemplate(template_str, name=name)
    return templates
