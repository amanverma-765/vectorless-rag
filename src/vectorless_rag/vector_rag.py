import os
from pathlib import Path

import chromadb
from pydantic_ai import Embedder, EmbeddingSettings
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.providers.openai import OpenAIProvider

from vectorless_rag.loader import hash_pdf, load_hashes, load_pdf, save_hashes

HASHES_PATH = Path("chroma_db") / "hashes.json"

# 3072 dims, delete chroma_db/ and re-ingest if this changes
MODEL = "gemini-embedding-001"
provider = OpenAIProvider(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["GEMINI_API_KEY"],
)
embedding_model = OpenAIEmbeddingModel(
    MODEL,
    provider=provider,
    # sdk defaults to base64
    settings=EmbeddingSettings(extra_body={"encoding_format": "float"}),
)
embedder = Embedder(embedding_model)

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="documents")


def create_chunks(pages: list, size: int = 1000, overlap: int = 250) -> list[dict]:
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be less than size ({size})")

    # track which slice of doc came from which page
    doc = ""
    page_ranges = []

    for page_number, page_text in pages:
        text = " ".join(page_text.split())

        start = len(doc)
        doc += text + " "
        end = len(doc)

        page_ranges.append({
            "page": page_number,
            "start": start,
            "end": end,
        })

    chunks = []

    start = 0
    chunk_id = 0

    while start < len(doc):
        end = min(start + size, len(doc))
        chunked_text = doc[start:end].strip()

        # ranges tile the doc, always one match
        page = next(
            r["page"] for r in page_ranges if r["start"] <= start < r["end"]
        )

        chunks.append({
            "chunk_id": chunk_id,
            "page": page,
            "text": chunked_text,
        })

        chunk_id += 1

        if end >= len(doc):
            break

        start = end - overlap

    return chunks


async def create_embeddings(chunks: list[dict], batch: int = 100):
    texts = [chunk["text"] for chunk in chunks]

    # pydantic-ai sends the whole list in one request, too big for a long pdf
    embeddings = []
    for start in range(0, len(texts), batch):
        result = await embedder.embed_documents(texts[start:start + batch])
        embeddings.extend(result.embeddings)

    return [
        {
            **chunk,
            "embedding": embedding,
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]


def store_embeddings(chunks: list[dict], doc_id: str) -> None:
    # add() ignores dupe ids, upsert() leaves stale chunks behind. wipe first.
    collection.delete(where={"doc": doc_id})

    collection.add(
        ids=[
            f"{doc_id}-{chunk['chunk_id']}"
            for chunk in chunks
        ],
        documents=[
            chunk["text"]
            for chunk in chunks
        ],
        embeddings=[
            chunk["embedding"]
            for chunk in chunks
        ],
        metadatas=[
            {
                "doc": doc_id,  # what delete() matches on
                "page": chunk["page"],
            }
            for chunk in chunks
        ],
    )


async def ingest(pdf_path: str) -> None:
    doc_id = Path(pdf_path).stem
    digest = hash_pdf(pdf_path)
    hashes = load_hashes(HASHES_PATH)

    if hashes.get(doc_id) == digest:
        print("Already ingested, unchanged. Skipping.")
        return

    print("Loading PDF...")
    pages = load_pdf(pdf_path)

    print("Creating chunks...")
    chunks = create_chunks(pages)

    print(f"Created {len(chunks)} chunks")

    print("Creating embeddings...")
    chunks = await create_embeddings(chunks)

    print("Storing in Chroma...")
    # same-named pdfs in different dirs collide
    store_embeddings(chunks, doc_id)

    hashes[doc_id] = digest
    save_hashes(HASHES_PATH, hashes)

    print("Done!")


async def retrieve(question: str, k: int = 5) -> list[dict]:
    """Search the indexed document for passages relevant to a question.

    Args:
        question: What to search for. Reword it if the first search misses.
        k: How many passages to return.
    """
    print("Retrieving vector for: ", question)
    # embed_query, not embed_documents. gemini uses different task types.
    [vec] = await embedder.embed_query([question])
    res = collection.query(query_embeddings=[vec], n_results=k)

    # chroma nests one list per query
    return [
        {"text": text, "page": metadata["page"], "distance": distance}
        for text, metadata, distance in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        )
    ]
