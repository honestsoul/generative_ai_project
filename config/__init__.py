"""Configuration module for Generative AI Project.

Loads settings from environment variables and YAML config files.
API keys and secrets are NEVER stored in config files — they must be
provided via environment variables or a .env file.
"""

import os
import yaml
from pathlib import Path


CONFIG_DIR = Path(__file__).parent


def _load_yaml(filename: str) -> dict:
    """Load a YAML config file from the config directory."""
    filepath = CONFIG_DIR / filename
    if not filepath.exists():
        return {}
    with open(filepath, "r") as f:
        content = yaml.safe_load(f)
    return content if content else {}


def get_api_key(provider: str) -> str:
    """Retrieve an API key from environment variables.

    Args:
        provider: The LLM provider name (e.g., 'openai', 'anthropic').

    Returns:
        The API key string.

    Raises:
        EnvironmentError: If the required API key is not set.
    """
    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    env_var = env_var_map.get(provider.lower())
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}")

    api_key = os.environ.get(env_var)
    if not api_key:
        raise EnvironmentError(
            f"API key not found. Set the {env_var} environment variable. "
            f"See .env.example for reference."
        )
    return api_key


def get_model_config() -> dict:
    """Load model configuration (non-sensitive parameters only)."""
    return _load_yaml("model_config.yaml")


def get_prompt_templates() -> dict:
    """Load prompt templates configuration."""
    return _load_yaml("prompt_templates.yaml")


def get_logging_config() -> dict:
    """Load logging configuration."""
    return _load_yaml("logging_config.yaml")
