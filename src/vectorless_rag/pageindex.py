import asyncio
import json
from pathlib import Path

from pydantic_ai import Agent

from vectorless_rag.agent import model
from vectorless_rag.loader import hash_pdf, load_hashes, load_pdf, save_hashes

INDEX_DIR = Path("page_index")
HASHES_PATH = INDEX_DIR / "hashes.json"

# running total of what the selector has spent this process
selector_usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}

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
    doc_id = Path(pdf_path).stem
    digest = hash_pdf(pdf_path)
    hashes = load_hashes(HASHES_PATH)

    if hashes.get(doc_id) == digest:
        print("Already ingested, unchanged. Skipping.")
        return

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
    path = INDEX_DIR / f"{doc_id}.json"
    path.write_text(json.dumps(index, indent=2))

    hashes[doc_id] = digest
    save_hashes(HASHES_PATH, hashes)

    print(f"Done! Index at {path}")


def load_index() -> dict[int, dict]:
    """Every indexed page across all ingested PDFs, keyed by page number"""
    files = sorted(f for f in INDEX_DIR.glob("*.json") if f != HASHES_PATH)
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
    for key in selector_usage:
        selector_usage[key] += getattr(picked.usage, key) or 0

    # model can name a page that isn't there
    pages = [
        {"text": index[n]["text"], "page": n}
        for n in picked.output[:k]
        if n in index
    ]
    # an empty tool result makes the gemini backend reject the next request
    if not pages:
        return [{"text": "No relevant pages found.", "page": None}]
    return pages
