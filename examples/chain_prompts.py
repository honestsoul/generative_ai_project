"""Example: prompt chaining.

Demonstrates :class:`PromptChain` — each step's output is automatically
fed into the next step's template through the shared chaining utility.
No generation or templating logic is duplicated in calling code.
"""

from src.llm.utils import create_client
from src.prompt_engineering.chain import PromptChain
from src.prompt_engineering.templates import PromptTemplate
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    client = create_client(provider="openai", model="gpt-3.5-turbo")

    chain = PromptChain(client=client)

    chain.add_step(
        name="brainstorm",
        template=PromptTemplate(
            name="brainstorm",
            template="List 3 creative project ideas about {topic}.",
        ),
        output_key="ideas",
    )
    chain.add_step(
        name="evaluate",
        template=PromptTemplate(
            name="evaluate",
            template=(
                "Given these project ideas:\n{ideas}\n\n"
                "Pick the most feasible one and explain why in two sentences."
            ),
        ),
        output_key="evaluation",
    )
    chain.add_step(
        name="plan",
        template=PromptTemplate(
            name="plan",
            template=(
                "Based on this evaluation:\n{evaluation}\n\n"
                "Create a brief 3-step implementation plan."
            ),
        ),
        output_key="plan",
    )

    results = chain.run(initial_vars={"topic": "generative AI for education"})

    for step_result in results:
        print(f"\n--- {step_result['step'].upper()} ---")
        print(step_result["response"]["text"])

    print(f"\n--- FINAL OUTPUT ---\n{chain.get_final_output()}")


if __name__ == "__main__":
    main()
