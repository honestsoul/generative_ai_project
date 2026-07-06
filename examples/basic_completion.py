"""Example: basic text completion using the shared LLM client factory.

Demonstrates that callers never duplicate config loading, rate limiting,
caching, or error handling — all of that lives in the shared utilities.
"""

from src.llm.utils import create_client, select_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    suggestion = select_model(task="general")
    logger.info("Using %s / %s", suggestion["provider"], suggestion["model"])

    client = create_client(
        provider=suggestion["provider"],
        model=suggestion["model"],
    )

    result = client.generate(
        prompt="Explain the concept of prompt engineering in three sentences.",
        system_prompt="You are a helpful AI tutor.",
    )

    print(f"\nModel : {result['model']}")
    print(f"Tokens: {result['input_tokens']} in / {result['output_tokens']} out")
    if result.get("cost") is not None:
        print(f"Cost  : ${result['cost']:.6f}")
    print(f"\n{result['text']}")

    stats = client.get_usage_stats()
    logger.info("Session stats: %s", stats)


if __name__ == "__main__":
    main()
