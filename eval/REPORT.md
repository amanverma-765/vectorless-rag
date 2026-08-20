# Eval report: vector_rag vs pageindex

Full detail behind the [Eval results](../README.md#eval-results) section of
the README. Run date: 2026-08-21. Corpus: `data/doc/jesc101.pdf`, the NCERT
Class 10 Science chapter "Chemical Reactions and Equations", 16 pages. Both
arms answered the same 18 questions through the same agent
(`ag/gemini-3.7-flash-medium` via 9router); embeddings are
`gemini-embedding-001`. Reproduce with:

```bash
uv run python eval/run_eval.py               # vector arm
uv run python eval/run_eval.py --pageindex   # pageindex arm
uv run python eval/run_eval.py --report      # this comparison
```

## TL;DR

Pageindex wins on every quality metric (retrieval, rank, citations) at
2.7x the tokens and roughly double the latency, including one pathological
question that alone cost 196 seconds and 56k tokens. Neither arm
hallucinated on any of the three trap questions. Take the direction of
these results seriously and the magnitude with a grain of salt: it's one
run, one 16-page document, small enough to sit entirely in the selector's
prompt, which is exactly the regime that favors pageindex.

## The question set

18 questions, each written against the actual page text and labelled with
gold pages by hand. The categories target known failure modes of each arm:

- **chunk-boundary** (Q1, Q12): the evidence spans two pages, so a
  1000-char chunk can cut it in half.
- **cross-page equation gap** (Q2): an equation on one page, the sentence
  interpreting it on the next.
- **distractor disambiguation** (Q3): three same-topic pages, only one
  matching the constraint (electricity, not heat or light).
- **multi-hop** (Q4): the answer needs three pages at once.
- **paraphrase** (Q5): no lexical overlap with the source wording.
- **buried facts** (Q6, Q7, Q11): a one-sentence aside, a sidebar box, a
  rhetorical question, each on a page whose main topic is something else.
- **structured content** (Q13): the answer lives inside a PDF-extracted
  table.
- **semantic near-duplicate** (Q14): hydrogen burning to water vs water
  electrolysed back apart; nearly identical vocabulary, opposite reactions.
- **easy controls** (Q10, Q15): literal facts both arms should get; if
  these fail, the harness is broken, not the retriever.
- **negative traps** (Q8, Q9, Q18): plausible questions the chapter never
  answers. Gold is the empty set; the correct behavior is a grounded
  refusal. These are read by hand, not auto-scored.

## Headline numbers

| metric | vector_rag | pageindex |
| --- | --- | --- |
| Hit rate (gold page in top-5) | 0.93 | **1.00** |
| MRR | 0.88 | **0.93** |
| Mean recall of gold pages | 0.91 | **0.97** |
| Citation precision | 0.69 | **0.85** |
| Citation recall | 0.73 | **0.93** |
| Mean retrieve latency | **0.8 s** | 5.7 s |
| Mean answer latency | **12.0 s** | 25.8 s |
| Median answer latency | **10.5 s** | 12.7 s |
| Mean model requests per answer | **2.17** | 2.67 |
| Mean tokens per question | **~7.0k** | ~18.8k |
| Total tokens, full run | **125,773** | 337,984 |

The mean/median latency gap matters: pageindex's mean is dragged up by one
outlier (Q2, below). On a typical question it's ~12s vs ~11s at answer
time, plus ~5s of selector latency at retrieval time that the vector arm
spends as ~0.7s of embedding lookup.

## Per-question results

Hit is whether a gold page appears anywhere in the top-5 retrieved; rank is
the position of the first gold page in the ranked results. Negative-trap
rows (8, 9, 18) score neither, since gold is the empty set. `vec` =
vector_rag, `pi` = pageindex.

| id | category | gold | vec hit | vec rank | pi hit | pi rank |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | chunk-boundary | 4, 5 | yes | 1 | yes | 2 |
| 2 | cross-page equation gap | 12, 13 | yes | 1 | yes | 1 |
| 3 | distractor disambiguation | 9 | yes | 1 | yes | 1 |
| 4 | multi-hop, 3 pages | 8, 9, 10 | yes | 1 | yes | 1 |
| 5 | paraphrase | 9, 10 | yes | 1 | yes | 1 |
| 6 | buried mid-paragraph fact | 13 | yes | 1 | yes | 1 |
| 7 | buried sidebar box | 7 | yes | 1 | yes | 1 |
| 8 | negative trap | none | - | - | - | - |
| 9 | negative trap | none | - | - | - | - |
| 10 | easy control | 9 | yes | 1 | yes | 1 |
| 11 | buried rhetorical aside | 13 | yes | 1 | yes | 2 |
| 12 | chunk-boundary | 6, 7 | yes | 1 | yes | 1 |
| 13 | structured/tabular content | 4 | yes | 1 | yes | 1 |
| 14 | semantic near-duplicate | 7 | **no** | miss | yes | 1 |
| 15 | easy control | 5 | yes | 1 | yes | 1 |
| 16 | named-section lookup | 14 | yes | 4 | yes | 1 |
| 17 | page-internal disambiguation | 14 | yes | 1 | yes | 1 |
| 18 | negative trap | none | - | - | - | - |

vector_rag missed exactly once (Q14, a full write-up below). Its ranks are
otherwise all 1s except Q16, where the gold page was buried at rank 4.
pageindex hit every scored question, with rank 2 twice (Q1, Q11) where it
ranked a neighboring page ahead of the gold page without dropping it.

A structural difference explains most of the citation gap reported above:
the vector arm always returns 5 chunks whether they're relevant or not,
while the selector returns only the pages it believes in (usually 1-3
pages). The agent sees less noise from pageindex, so it cites less noise.

## Case studies

### Q14, the one clean win

"Which two elements react to directly form water as their only product,
and is that reaction combination or displacement?" The answer is hydrogen
plus oxygen on page 7. The vector arm retrieved pages 6, 11, and 14 and
never saw page 7: the electrolysis page and the displacement pages share
almost all their vocabulary (water, hydrogen, oxygen, reaction) with the
question, and cosine similarity has no way to notice that electrolysis is
the *reverse* reaction. The selector read the page 7 summary, understood
"formation of water from H2 and O2" is the thing being asked about, and
picked exactly that page. This is the "reasons about relevance instead of
measuring proximity" claim from the README, observed in the wild.

### Q2, the one pathological loss

"In equation 1.32, which substance is reduced?" The equation is on page
12; the sentence identifying MnO2 as the reduced substance is on page 13.
The selector picked only page 12 (a summary that says a page *contains
equations* doesn't say which equation numbers), so the agent got half the
evidence, reworded, retried, and looped: 7 model requests, 196 seconds
end to end, 56.7k tokens for one question, roughly a sixth of the arm's
entire run budget. It did land the right answer and cited page 13 in the end, but
this is the failure shape to know about: when the index's summaries don't
carry the discriminating detail, the page index doesn't fail fast, it
fails *expensively*. The vector arm got both pages instantly because
"equation 1.32" appears verbatim in a chunk.

### Q16, rank quality

"According to the chapter's summary section ('What you have learnt')...":
the vector arm's top 5 was pages 13, 15, 6, 14, 6 in that order, so the
gold page (14) came back at rank 4, behind three wrong pages that also
discuss decomposition and combination, and the agent then cited page 13
instead. The selector went straight to page 14 at rank 1 and the citation
followed. Retrieval rank isn't cosmetic; the agent tends to trust whatever
is at the top.

### Q13, right answer, wrong citation

Both arms retrieved the right page (4) for the atom-count table question.
The vector arm's answer cited page 3, which contains a *different*
balancing table (Zn + H2SO4) whose hydrogen row happens to read the same
(2 and 2). A plausible-looking citation to the wrong evidence is exactly
the kind of failure that citation precision exists to catch, and it's
worth noting it happened on tabular content, where chunked text loses the
table's visual identity.

### Q1 and Q11, where chunking quietly helped

The selector's only rank-2 results were on Q1 and Q11, where it also
picked a neighboring page (3 and 12 respectively) ahead of the gold page.
Harmless here, since the gold page was still included, but it shows the
selector's judgment is holistic rather than exact: it reasons about where
an answer *should* live, and adjacent pages often qualify.

## The negative traps

All three were answered correctly by both arms. Verbatim answers are in
the `--report` output; summarized:

- **Q8 (safety equipment for electrolysis at home):** both arms retrieved
  the electrolysis page, correctly said the chapter recommends nothing for
  home use, and quoted the only related caution (teacher handles the
  candle test). Both cited page 9 while refusing, which the precision
  metric counts against them; arguably citing the page you checked is the
  more useful behavior.
- **Q9 (boiling point of MgO):** both arms gave a flat "not mentioned"
  with no citation and no smuggled-in world knowledge. This was the purest
  test of grounding and both passed.
- **Q18 (which bonds are in Chapters 3 and 4):** the trap worked as
  designed: both arms retrieved the page that mentions "types of bonds...
  in Chapters 3 and 4" and both correctly reported that the chapter never
  names the bonds. Strong lexical match, no answer, no hallucination.

The pageindex arm's selector returned an empty pick for all three, which
is itself a signal the vector arm cannot produce: top-k always returns
something, so "nothing matches" has to be inferred from distances.

## Cost anatomy

Where the pageindex tokens actually go, per question on average:

| component | vector_rag | pageindex |
| --- | --- | --- |
| Answering agent input | 6,837 | 8,114 |
| Answering agent output | 151 | 157 |
| Selector (retrieval LLM) input | 0 | 10,464 |
| **Total per question** | **~7.0k** | **~18.8k** |

The selector is the whole story: every `retrieve` call puts all 16 page
summaries into a prompt, and the agent averages more than one retrieve per
question. On this document that's ~2.6k tokens per selector call. At 160
pages it would be ~26k per call, several calls per question. The vector
arm's retrieval cost is a single embedding call (fractions of a cent,
milliseconds) regardless of corpus size. This is the linear-vs-sublinear
divide the README describes, now with numbers attached.

The answering agent's input is also larger for pageindex (8.1k vs 6.8k)
because it returns whole pages rather than 1000-char chunks. That's the
same tradeoff seen from the other side: more context per passage is why
the citations are better, and it's also why it costs more.

## What running the eval caught beyond the numbers

The first pageindex run crashed reliably on Q8, and the cause turned out
to be a real product bug, not eval flakiness: when the selector picks no
pages, `retrieve` returned an empty list, and an empty tool result makes
the Gemini backend reject the follow-up request ("Requests ending with a
model turn are not supported"), after which the router holds the route in
a ~30s cooldown that fails everything behind it. Any user asking one
unanswerable question would have hit it. `retrieve` now returns a
"No relevant pages found." sentinel instead. An eval that only measured
happy paths would never have found this; the negative-trap questions
earned their place before a single metric was read.

Two harness bugs were also found and fixed during the run: the citation
regex missed plural forms ("pages 5, 6"), which had been silently
deflating citation scores for both arms, and the hash sidecar file was
initially swept up by the page-index loader's glob.

## Caveats

- **n = 1 run, one document, one model.** No variance estimates; the
  selector and agent are nondeterministic, so individual ranks could
  wobble across runs. The aggregate direction (pageindex better quality,
  vector cheaper) is consistent enough across 18 questions to trust; any
  single row is not.
- **The corpus is small and favorable to pageindex.** 16 pages fits
  comfortably in one selector prompt. The comparison at 300 pages would
  need the hierarchical index the README's roadmap describes, and these
  numbers say nothing about that regime.
- **Gold labels and questions were written by the same hand.** Standard
  small-eval bias; a second labeller would strengthen the labels.
- **Citation extraction is a regex** over "page N" patterns. It handles
  the observed formats but any exotic phrasing would slip through.
- **Rank semantics differ slightly between arms**: vector ranks are over
  the chunk list (which can repeat a page), pageindex ranks are over
  distinct pages. Hit rate and recall are unaffected; MRR comparisons
  across arms carry a small asymmetry.
- **Answer correctness is not auto-graded.** Hit/citation metrics proxy
  it; the negative traps were verified by reading. A judge model would be
  the next step if the question set grows.

## Verdict, and what to measure next

On this document the page index is simply the better retriever, and the
margin shows up exactly where the question set was designed to probe:
semantic reversal (Q14), rank quality (Q16), and citation discipline
across the board. The vector arm's wins are structural, not qualitative:
2.7x fewer tokens per question, retrieval that is both an order of
magnitude faster and nearly free (one embedding call against ~10k LLM
tokens of selector prompt), and no failure mode that costs 56k tokens.

The most interesting next measurements, in order:

1. **The hybrid** (embed the summaries, vector-shortlist, selector picks
   from the shortlist): the README's roadmap item 4. These results
   predict it keeps pageindex's quality at a fraction of the selector
   cost, which is now a testable claim.
2. **Richer summaries for discriminating detail**: Q2 failed because the
   summary didn't carry equation numbers. One prompt line ("include
   equation and figure numbers") and a re-ingest would show whether the
   fix is really that cheap.
3. **A larger document**, to watch the selector prompt cost curve bend
   from tolerable to prohibitive and find where the crossover actually
   is.
