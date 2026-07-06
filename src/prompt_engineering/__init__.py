"""Prompt engineering package.

Re-exports the main classes for convenient access::

    from src.prompt_engineering import PromptTemplate, FewShotPrompt, PromptChain
"""

from src.prompt_engineering.chain import PromptChain
from src.prompt_engineering.few_shot import FewShotPrompt
from src.prompt_engineering.templates import PromptTemplate, TemplateManager

__all__ = ["FewShotPrompt", "PromptChain", "PromptTemplate", "TemplateManager"]
