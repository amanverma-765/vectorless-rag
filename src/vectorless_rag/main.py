import asyncio
import sys

from dotenv import load_dotenv
load_dotenv()

from vectorless_rag import pageindex, vector_rag
from vectorless_rag.agent import agent, answer


def main() -> None:
    # argparse when there's a third one
    retriever = pageindex if "--pageindex" in sys.argv else vector_rag
    # registered here so agent.py stays retriever-agnostic
    agent.tool_plain(retriever.retrieve)
    print(f"Retriever: {retriever.__name__}")

    path = input("Paste path of the pdf doc: ")
    asyncio.run(retriever.ingest(path))
    print("=" * 100)
    while True:
        quest = input("user > ")
        if quest == "exit":
            break
        print(f'bot >  {asyncio.run(answer(quest))}')


if __name__ == "__main__":
    main()
