import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# local 9router proxy, falls back when a provider runs out of quota
MODEL = "ag/gemini-3.7-flash-medium"
provider = OpenAIProvider(
    base_url="http://localhost:20128/v1",
    api_key=os.environ["NINEROUTER_API_KEY"],
)
model = OpenAIChatModel(
    MODEL,
    provider=provider,
)
agent = Agent(
    model,
    instructions=(
        "Answer questions about the indexed document. "
        "Search again with reworded queries if the passages look insufficient. "
        "Answer concisely, grounded only in retrieved passages, and cite the "
        "page number for each claim. Cite only pages the tool actually "
        "returned. If the passages do not contain the answer, say so."
    ),
)


async def answer(question: str) -> str:
    result = await agent.run(question)

    return result.output
