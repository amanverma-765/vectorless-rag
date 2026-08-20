import asyncio
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from vectorless_rag import pageindex, vector_rag
from vectorless_rag.agent import agent

RESULTS_DIR = Path(__file__).parent

# gold_pages: [] means the question has no answer in the document; correct
# behavior is refusing to answer, not a retrieval hit.
QUESTIONS = [
    {"id": 1, "category": "chunk-boundary", "gold_pages": [4, 5],
     "question": "Walk through the hit-and-trial method for balancing the reaction between "
                 "iron and steam, and name which element should be balanced first."},
    {"id": 2, "category": "cross-page equation gap", "gold_pages": [12, 13],
     "question": "In equation 1.32, which substance is reduced?"},
    {"id": 3, "category": "distractor disambiguation", "gold_pages": [9],
     "question": "Which specific decomposition reaction uses electricity, not heat or light, "
                 "and what does the experiment demonstrate?"},
    {"id": 4, "category": "multi-hop, 3 pages", "gold_pages": [8, 9, 10],
     "question": "List the three different forms of energy that can drive a decomposition "
                 "reaction, with one example of each."},
    {"id": 5, "category": "paraphrase, no literal overlap", "gold_pages": [9, 10],
     "question": "How do you visually tell that silver-based photography chemicals have "
                 "reacted, without touching or heating anything?"},
    {"id": 6, "category": "buried mid-paragraph fact", "gold_pages": [13],
     "question": "According to the chapter, what everyday trick do potato-chip makers use "
                 "to stop their product from going rancid?"},
    {"id": 7, "category": "buried sidebar box", "gold_pages": [7],
     "question": "Per the textbook's 'Do You Know?' box, what everyday building material is "
                 "chemically identical to a compound formed by calcium hydroxide reacting "
                 "with air, and how long does it take to form?"},
    {"id": 8, "category": "negative, plausible trap", "gold_pages": [],
     "question": "What safety equipment does the chapter recommend when performing "
                 "electrolysis of water at home?"},
    {"id": 9, "category": "negative, plausible trap", "gold_pages": [],
     "question": "What is the boiling point of magnesium oxide, according to the chapter?"},
    {"id": 10, "category": "easy control", "gold_pages": [9],
     "question": "What voltage battery is used in the water electrolysis setup?"},
    {"id": 11, "category": "buried rhetorical aside", "gold_pages": [13],
     "question": "The chapter poses a question mid-page asking the reader to identify "
                 "whether burning magnesium in air is oxidation or reduction. Which page "
                 "asks this?"},
    {"id": 12, "category": "chunk-boundary", "gold_pages": [6, 7],
     "question": "Name a combination reaction from the chapter that is also explicitly "
                 "labelled exothermic, and give its balanced equation."},
    {"id": 13, "category": "structured/tabular content", "gold_pages": [4],
     "question": "In the worked balancing example for iron and steam, how many hydrogen "
                 "atoms are on each side of the equation before balancing (the very first "
                 "count)?"},
    {"id": 14, "category": "semantic near-duplicate trap", "gold_pages": [7],
     "question": "Which two elements react to directly form water as their only product, "
                 "and is that reaction combination or displacement?"},
    {"id": 15, "category": "easy control", "gold_pages": [5],
     "question": "What do the letters (s), (l), (g), and (aq) mean next to a chemical "
                 "formula?"},
    {"id": 16, "category": "named-section lookup", "gold_pages": [14],
     "question": "According to the chapter's summary section ('What you have learnt'), "
                 "what's the relationship between decomposition and combination reactions?"},
    {"id": 17, "category": "page-internal disambiguation", "gold_pages": [14],
     "question": "In the textbook's end-of-chapter exercises, which reaction (by reactants) "
                 "tests understanding of redox with lead oxide and carbon?"},
    {"id": 18, "category": "strong-lexical-match trap", "gold_pages": [],
     "question": "What specific chemical bonds are discussed in Chapters 3 and 4, according "
                 "to this chapter?"},
]


def cited_pages_in(text: str) -> list[int]:
    # models write "page 9", "p. 9", "pg 9", "pages 5, 6", "pages 5 and 6"
    pages = set()
    for group in re.findall(
        r"\bp(?:age|g)?s?\.?\s*(\d+(?:\s*(?:,|and|&)\s*\d+)*)", text, re.IGNORECASE
    ):
        pages.update(int(n) for n in re.findall(r"\d+", group))
    return sorted(pages)


def score(gold: list[int], retrieved: list[int], cited: list[int]) -> dict:
    gold_set = set(gold)
    cited_set = set(cited)
    return {
        "hit": bool(gold_set & set(retrieved)) if gold_set else None,
        "rank": next((i + 1 for i, p in enumerate(retrieved) if p in gold_set), None) if gold_set else None,
        "recall": len(gold_set & set(retrieved)) / len(gold_set) if gold_set else None,
        "citation_precision": len(gold_set & cited_set) / len(cited_set) if cited_set else None,
        "citation_recall": len(gold_set & cited_set) / len(gold_set) if gold_set else None,
    }


def selector_delta(retriever, snap: dict) -> dict:
    # tokens the pageindex selector spent since snap; vector_rag has no selector
    usage = getattr(retriever, "selector_usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "requests": 0}
    return {key: usage[key] - snap.get(key, 0) for key in usage}


def selector_snapshot(retriever) -> dict:
    return dict(getattr(retriever, "selector_usage", {}))


async def eval_question(retriever, q: dict) -> dict:
    snap = selector_snapshot(retriever)
    t0 = time.perf_counter()
    retrieved = await retriever.retrieve(q["question"], k=5)
    retrieve_s = time.perf_counter() - t0
    retrieve_selector = selector_delta(retriever, snap)
    retrieved_pages = [r["page"] for r in retrieved if r.get("page") is not None]

    snap = selector_snapshot(retriever)
    t0 = time.perf_counter()
    run = await agent.run(q["question"])
    answer_s = time.perf_counter() - t0
    tool_selector = selector_delta(retriever, snap)
    answer_text = run.output
    usage = run.usage
    cited = cited_pages_in(answer_text)

    answer_input = usage.input_tokens or 0
    answer_output = usage.output_tokens or 0
    return {
        **q,
        "retrieved_pages": retrieved_pages,
        "cited_pages": cited,
        "retrieve_latency_s": round(retrieve_s, 2),
        "answer_latency_s": round(answer_s, 2),
        "answer_requests": usage.requests,
        "answer_input_tokens": answer_input,
        "answer_output_tokens": answer_output,
        "selector_retrieve_input_tokens": retrieve_selector["input_tokens"],
        "selector_retrieve_output_tokens": retrieve_selector["output_tokens"],
        "selector_answer_input_tokens": tool_selector["input_tokens"],
        "selector_answer_output_tokens": tool_selector["output_tokens"],
        "total_tokens": answer_input + answer_output
                        + retrieve_selector["input_tokens"] + retrieve_selector["output_tokens"]
                        + tool_selector["input_tokens"] + tool_selector["output_tokens"],
        "answer_text": answer_text,
        **score(q["gold_pages"], retrieved_pages, cited),
    }


async def run(retriever) -> list[dict]:
    records = []
    for q in QUESTIONS:
        for attempt in (1, 2):
            try:
                record = await eval_question(retriever, q)
                print(f"  {q['id']:>2}. {q["category"]:<32} hit={record['hit']!s:<5} "
                      f"rank={record['rank']!s:<5} tokens={record['total_tokens']}")
                break
            except Exception as e:  # noqa: BLE001 - one bad question shouldn't drop the whole run
                if attempt == 1:
                    # 9router holds a failing route in cooldown for ~30s
                    print(f"  {q['id']:>2}. error, retrying in 35s: {e}")
                    await asyncio.sleep(35)
                else:
                    record = {**q, "error": str(e)}
                    print(f"  {q['id']:>2}. {q["category"]:<32} ERROR: {e}")
        records.append(record)

    return records


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def aggregate(records: list[dict]) -> dict:
    records = [r for r in records if "error" not in r]
    scored = [r for r in records if r["gold_pages"]]
    ranks = [r["rank"] for r in scored if r["rank"]]
    return {
        "hit_rate": mean(r["hit"] for r in scored),
        "mrr": sum(1 / r for r in ranks) / len(scored) if scored else None,
        "mean_recall": mean(r["recall"] for r in scored),
        "mean_citation_precision": mean(r["citation_precision"] for r in records),
        "mean_citation_recall": mean(r["citation_recall"] for r in scored),
        "mean_retrieve_latency_s": mean(r["retrieve_latency_s"] for r in records),
        "mean_answer_latency_s": mean(r["answer_latency_s"] for r in records),
        "mean_answer_requests": mean(r["answer_requests"] for r in records),
        "mean_answer_input_tokens": mean(r["answer_input_tokens"] for r in records),
        "mean_answer_output_tokens": mean(r["answer_output_tokens"] for r in records),
        "mean_selector_input_tokens": mean(r["selector_retrieve_input_tokens"]
                                           + r["selector_answer_input_tokens"] for r in records),
        "mean_tokens_per_question": mean(r["total_tokens"] for r in records),
        "total_tokens": sum(r["total_tokens"] for r in records),
    }


def fmt(x) -> str:
    return f"{x:.2f}" if isinstance(x, float) else str(x)


def report() -> None:
    results = {}
    for name in ("vector_rag", "pageindex"):
        path = RESULTS_DIR / f"results_{name}.json"
        if not path.exists():
            sys.exit(f"missing {path}, run 'uv run python eval/run_eval.py"
                      f"{' --pageindex' if name == 'pageindex' else ''}' first")
        results[name] = json.loads(path.read_text())

    aggregates = {name: aggregate(records) for name, records in results.items()}

    print(f"{'metric':<30}{'vector_rag':>14}{'pageindex':>14}")
    for key in aggregates["vector_rag"]:
        print(f"{key:<30}{fmt(aggregates['vector_rag'][key]):>14}{fmt(aggregates['pageindex'][key]):>14}")

    print(f"\n{'id':<4}{"category":<32}{'vec hit':<9}{'vec rank':<10}{'pi hit':<8}{'pi rank':<8}")
    for v, p in zip(results["vector_rag"], results["pageindex"]):
        print(f"{v['id']:<4}{v["category"]:<32}{fmt(v.get('hit')):<9}{fmt(v.get('rank')):<10}"
              f"{fmt(p.get('hit')):<8}{fmt(p.get('rank')):<8}")

    errored = [(v, p) for v, p in zip(results["vector_rag"], results["pageindex"])
               if "error" in v or "error" in p]
    if errored:
        print("\n--- errored (excluded from aggregates above) ---")
        for v, p in errored:
            print(f"Q{v['id']}: vector_rag={v.get('error', 'ok')} pageindex={p.get('error', 'ok')}")

    print("\n--- negative questions: read for hallucination, not auto-scored ---")
    for v, p in zip(results["vector_rag"], results["pageindex"]):
        if v["gold_pages"] or "error" in v or "error" in p:
            continue
        print(f"\nQ{v['id']}: {v['question']}")
        print(f"  vector_rag cited={v['cited_pages']}: {v['answer_text']}")
        print(f"  pageindex  cited={p['cited_pages']}: {p['answer_text']}")


def main() -> None:
    if "--report" in sys.argv:
        report()
        return

    retriever = pageindex if "--pageindex" in sys.argv else vector_rag
    agent.tool_plain(retriever.retrieve)

    name = retriever.__name__.rsplit(".", 1)[-1]
    print(f"Evaluating {name} on {len(QUESTIONS)} questions...")
    records = asyncio.run(run(retriever))

    out = RESULTS_DIR / f"results_{name}.json"
    out.write_text(json.dumps(records, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
