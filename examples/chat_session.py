"""Example: multi-turn chat session.

Builds on the shared :func:`create_client` factory and
:func:`format_chat_messages` helper so that conversation history
management is consistent and non-duplicated.
"""

from src.llm.utils import create_client, format_chat_messages
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = "You are a knowledgeable assistant specializing in generative AI."


def run_chat() -> None:
    client = create_client(provider="openai", model="gpt-3.5-turbo")
    conversation: list[dict[str, str]] = []

    print("Chat session started (type 'quit' to exit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break

        conversation.append({"role": "user", "content": user_input})

        prompt = "\n".join(
            f"{m['role'].title()}: {m['content']}"
            for m in format_chat_messages(conversation, SYSTEM_PROMPT)
        )

        result = client.generate(prompt, system_prompt=SYSTEM_PROMPT)
        assistant_text = result["text"]

        conversation.append({"role": "assistant", "content": assistant_text})
        print(f"Assistant: {assistant_text}\n")

    stats = client.get_usage_stats()
    print(f"\nSession stats: {stats}")


if __name__ == "__main__":
    run_chat()
