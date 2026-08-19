import asyncio

from dotenv import load_dotenv
load_dotenv()  # ponytail: walks up from this file to the repo-root .env

from vectorless_rag.baseline import ingest, answer



def main() -> None:
    path = input("Paste path of the pdf doc: ")
    asyncio.run(ingest(path))
    print("=" * 100)
    while True:
        quest = input("user > ")
        if quest == "exit":
            break
        print(f'bot >  {asyncio.run(answer(quest))}')


if __name__ == "__main__":
    main()
