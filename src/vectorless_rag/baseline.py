import os
from pathlib import Path

import chromadb
from pydantic_ai import Embedder, EmbeddingSettings
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.providers.openai import OpenAIProvider

from vectorless_rag.loader import load_pdf

# Embeddings setup
# ponytail: Gemini speaks OpenAI, so the same provider class works. 3072 dims --
# delete chroma_db/ and re-ingest if you ever switch models again.
MODEL = "gemini-embedding-001"
provider = OpenAIProvider(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["GEMINI_API_KEY"],
)
model = OpenAIEmbeddingModel(
    MODEL,
    provider=provider,
    # ponytail: openai-sdk defaults encoding_format=base64; ask for floats
    settings=EmbeddingSettings(extra_body={"encoding_format": "float"}),
)
embedder = Embedder(model)

# Chroma setup
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="documents"
)


def create_chunks(
        pages: list,
        size: int = 1000,
        overlap: int = 250
) -> list[dict]:
    # Without this the chunk loop never advances and spins forever
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be less than size ({size})")

    # Combine text while keeping track of which page each character belongs to
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

        # Find all pages touched by this chunk
        chunk_pages = [
            page_info["page"]
            for page_info in page_ranges
            if page_info["start"] < end and page_info["end"] > start
        ]

        chunks.append({
            "chunk_id": chunk_id,
            # page_ranges tile the doc, so a chunk always touches one
            "page": chunk_pages[0],
            "pages": chunk_pages,
            "text": chunked_text,
        })

        chunk_id += 1

        if end >= len(doc):
            break

        start = end - overlap

    return chunks


async def create_embeddings(chunks: list[dict], batch: int = 100):
    texts = [chunk["text"] for chunk in chunks]

    # ponytail: pydantic-ai sends the whole list in one request, so batch here
    # or a large PDF blows the request limit. Sequential; parallelise if slow.
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
    # Clear the doc first: add() silently drops duplicate ids, and upsert()
    # would leave stale chunks behind when re-chunking yields fewer of them
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
                "doc": doc_id,  # what the delete above matches on
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "pages": ",".join(map(str, chunk["pages"])),
            }
            for chunk in chunks
        ],
    )

async def ingest(pdf_path: str) -> None:
    print("Loading PDF...")
    pages = load_pdf(pdf_path)

    print("Creating chunks...")
    chunks = create_chunks(pages)

    print(f"Created {len(chunks)} chunks")

    print("Creating embeddings...")
    chunks = await create_embeddings(chunks)

    print("Storing in Chroma...")
    # ponytail: stem collides for same-named PDFs in different dirs; hash the
    # full path if that ever happens
    store_embeddings(chunks, Path(pdf_path).stem)

    print("Done!")

async def retrieve(question: str, k: int = 5) -> list[dict]:
    """Search the indexed document for passages relevant to a question.

    Args:
        question: What to search for. Rephrase the user's question if it helps.
        k: How many passages to return.
    """
    print("Retrieving vector for: ", question)
    # ponytail: embed_query, not embed_documents -- nemotron-embed is asymmetric
    [vec] = await embedder.embed_query([question])
    res = collection.query(query_embeddings=[vec], n_results=k)

    # Chroma nests one list per query; we only ever send one
    return [
        {"text": text, "page": metadata["page"], "distance": distance}
        for text, metadata, distance in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        )
    ]
