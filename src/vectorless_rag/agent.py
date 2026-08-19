import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
provider = OpenAIProvider(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
model = OpenAIChatModel(
    MODEL,
    provider=provider,
)
agent = Agent(
    model,
    instructions=(
        "Answer questions about the indexed document. "
        "Always call the retrieve tool first -- never answer from memory. "
        "Search again with reworded queries if the passages look insufficient. "
        "Answer concisely, grounded only in retrieved passages, and cite the "
        "page number for each claim. Cite only pages the tool actually "
        "returned. If the passages do not contain the answer, say so."
    ),
)
