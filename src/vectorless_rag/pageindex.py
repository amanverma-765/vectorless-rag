async def ingest(pdf_path: str) -> None:
    raise NotImplementedError("pageindex ingest")


async def retrieve(question: str, k: int = 5) -> list[dict]:
    """Search the indexed document for passages relevant to a question.

    Args:
        question: What to search for. Rephrase the user's question if it helps.
        k: How many passages to return.
    """
    print("Retrieving page index...")
    raise NotImplementedError("pageindex retrieve")
