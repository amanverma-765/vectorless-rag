import asyncio
import json
from pathlib import Path

from pydantic_ai import Agent

from vectorless_rag.agent import model
from vectorless_rag.loader import load_pdf

INDEX_DIR = Path("page_index")

summarizer = Agent(
    model,
    instructions=(
        "One sentence: what would a reader find on this page? "
        "Name the specific topics, terms and examples covered. "
        "Write only the summary."
    )
)
selector = Agent(
    model,
    output_type=list[int],
    instructions=(
        "You are given a page index: one summary per page of a document. "
        "Return the page numbers that answer the question, fewest first. "
        "Return an empty list if none of them fit."
    )
)


async def ingest(pdf_path: str) -> None:
    print("Ingesting PDF...")
    # blank pages waste a call
    pages = [(n, t) for n, t in load_pdf(pdf_path) if t.strip()]

    print(f"Summarizing {len(pages)} pages...")
    # one call per page, all at once. add a semaphore if rpm becomes an issue
    runs = await asyncio.gather(*(summarizer.run(text) for _, text in pages))

    # keep the text here so retrieve doesn't reopen the pdf
    index = [
        {"page": n, "summary": r.output, "text": t}
        for (n, t), r in zip(pages, runs)
    ]

    INDEX_DIR.mkdir(exist_ok=True)
    path = INDEX_DIR / f"{Path(pdf_path).stem}.json"
    path.write_text(json.dumps(index, indent=2))

    print(f"Done! Index at {path}")


def load_index() -> dict[int, dict]:
    """Every indexed page across all ingested PDFs, keyed by page number"""
    files = sorted(INDEX_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no page index in {INDEX_DIR}/ -- ingest a PDF first")

    return {
        page["page"]: page
        for file in files
        for page in json.loads(file.read_text())
    }


async def retrieve(question: str, k: int = 5) -> list[dict]:
    """Search the indexed document for passages relevant to a question.

    Args:
        question: What to search for. Reword it if the first search misses.
        k: How many passages to return.
    """
    print("Retrieving page index for: ", question)
    index = load_index()

    summaries = "\n".join(
        f"page {n}: {page['summary']}"
        for n, page in sorted(index.items())
    )
    picked = await selector.run(
        f"{summaries}\n\nQuestion: {question}\nPick at most {k} pages."
    )

    # model can name a page that isn't there
    return [
        {"text": index[n]["text"], "page": n}
        for n in picked.output[:k]
        if n in index
    ]
