import asyncio
import sys

from dotenv import load_dotenv
load_dotenv()  # ponytail: walks up from this file to the repo-root .env

from vectorless_rag import baseline, pageindex
from vectorless_rag.agent import agent, answer


def main() -> None:
    # ponytail: one flag, two retrievers; argparse when there's a third option
    retriever = pageindex if "--vectorless" in sys.argv else baseline
    # registered here, not in agent.py, so agent.py stays retriever-agnostic
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
