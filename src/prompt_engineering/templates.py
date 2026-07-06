"""Prompt template management.

Loads templates from the shared YAML config so that prompt definitions
are not scattered across client code or example scripts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TEMPLATES_PATH = Path("config/prompt_templates.yaml")


class PromptTemplate:
    """A single reusable prompt template with variable substitution.

    Args:
        name: Human-readable template name.
        template: The template string with ``{variable}`` placeholders.
        description: Optional description of what the template does.
    """

    def __init__(
        self,
        name: str,
        template: str,
        description: str = "",
    ) -> None:
        self.name = name
        self.template = template
        self.description = description
        self.variables = self._extract_variables()

    def _extract_variables(self) -> list[str]:
        return re.findall(r"\{(\w+)\}", self.template)

    def render(self, **kwargs: str) -> str:
        """Fill in the template with the given keyword arguments.

        Raises :class:`ValueError` if required variables are missing.
        """
        missing = set(self.variables) - set(kwargs)
        if missing:
            raise ValueError(
                f"Template {self.name!r} missing variables: {missing}"
            )
        return self.template.format(**kwargs)


class TemplateManager:
    """Registry of :class:`PromptTemplate` objects, optionally loaded from YAML.

    Args:
        templates_path: Path to the YAML file containing template definitions.
    """

    def __init__(
        self, templates_path: Path = DEFAULT_TEMPLATES_PATH
    ) -> None:
        self._templates: dict[str, PromptTemplate] = {}
        self._load_from_yaml(templates_path)

    def _load_from_yaml(self, path: Path) -> None:
        if not path.exists():
            logger.info("No templates file at %s, starting empty", path)
            return
        with open(path) as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}
        for name, info in data.get("templates", {}).items():
            self.register(
                PromptTemplate(
                    name=name,
                    template=info["template"],
                    description=info.get("description", ""),
                )
            )
        logger.info("Loaded %d templates from %s", len(self._templates), path)

    def register(self, template: PromptTemplate) -> None:
        """Add or replace a template in the registry."""
        self._templates[template.name] = template

    def get(self, name: str) -> Optional[PromptTemplate]:
        """Look up a template by name."""
        return self._templates.get(name)

    def render(self, name: str, **kwargs: str) -> str:
        """Shorthand: look up *name* and render it."""
        tpl = self.get(name)
        if tpl is None:
            raise KeyError(f"Unknown template: {name!r}")
        return tpl.render(**kwargs)

    def list_templates(self) -> list[str]:
        """Return sorted list of registered template names."""
        return sorted(self._templates)
