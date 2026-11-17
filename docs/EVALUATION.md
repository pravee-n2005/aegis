# M5 Evaluation Suite

A deterministic, repeatable evaluation layer for AEGIS's RAG retrieval and
citation behavior, built as part of M5. It lives in `backend/eval/` and is
wired into the standard offline test suite via `backend/tests/test_eval.py`.

## How to run it

```bash
cd backend
source .venv/bin/activate

# Standalone report (human-readable table)
python -m eval.run_eval

# Machine-readable
python -m eval.run_eval --json

# Different Recall@K/Precision@K cutoff
python -m eval.run_eval --top-k 5

# As part of the regular test suite (this IS part of `pytest -q`)
pytest -q tests/test_eval.py
```

Both paths are fully offline and deterministic: no network access, no
`OPENROUTER_API_KEY` is actually used, no `sentence-transformers` model is
downloaded or run, and no real LLM call is made. This is intentional (see
"Deterministic vs. model-dependent" below) and is why `pytest -q` as a
whole never depends on a paid API call or model availability.

## What it measures

The suite has two parts:

### 1. Retrieval evaluation (`eval.harness.run_retrieval_eval`)

Runs every question in the fixed dataset (`eval/dataset.py`) through the
**real** `app.retrieval.search()` function against a fixed, hand-labeled
corpus (`eval/corpus.py`), and scores the results against each question's
ground-truth relevant `(filename, page)` set using:

- **Recall@K** — fraction of the relevant pages that appear in the top-K
  retrieved results. Measures "did we find everything relevant".
- **Precision@K** — fraction of the top-K retrieved results that are
  actually relevant. Measures "how much noise came with it".
- **MRR (Mean Reciprocal Rank)** — averages `1/rank` of the first relevant
  result across all questions. Measures how high up the relevant result
  tends to rank, not just whether it's present somewhere in the top-K.

It also checks that every citation retrieval returns (`filename`, `page`)
genuinely exists in the corpus — catching a retrieval-layer bug that
invents or mismatches a citation, independent of relevance.

### 2. Grounded-answer / citation evaluation (`eval.harness.run_grounded_answer_eval`)

Drives the **real** agent loop (`app.agent.ask_aegis`) end to end — real
`search_documents` tool dispatch, real `retrieval.search()`, real
`rounds_remaining`/`new_chunk_count` plumbing (the M5 reliability fix) —
with a **scripted** LLM client that calls `search_documents` once and then
answers citing the top real relevant page. This proves the
`search_documents → tool_result → citation` pipeline round-trips correct
citation metadata, without a real, paid LLM call.

## Metrics, dataset size, and methodology

- **Corpus**: 9 hand-authored chunks (one page each) across 2 documents
  (`handbook.pdf`, 5 pages; `jd.pdf`, 4 pages) — see `eval/corpus.py`.
- **Dataset**: 6 questions, each with a hand-labeled `relevant` set of
  `(filename, page)` pairs — see `eval/dataset.py`.
- **Embeddings**: NOT the real `sentence-transformers/all-MiniLM-L6-v2`
  model. Each chunk and each dataset query is assigned a hand-crafted
  "topic vector" (`corpus.topic_vector`) along 8 orthogonal topic axes, so
  cosine-similarity ranking is exact and reproducible without a model
  download. One chunk (`handbook.pdf` page 5) deliberately blends two
  topics (security + expense) to exercise discrimination — a query for
  either topic alone legitimately also retrieves it, which is why
  Precision@K is **not** expected to be 1.0 for every case
  (`test_retrieval_eval_precision_reflects_topic_overlap_not_perfect`
  pins this as known, intentional behavior rather than a bug).
- **Default top_k**: 3 (configurable via `--top-k`).

Current result on this fixed corpus/dataset (reproducible by running
`python -m eval.run_eval`): **mean Recall@K = 1.0, MRR = 1.0, mean
Precision@K ≈ 0.86** (less than 1.0 specifically because of the
intentional multi-topic chunk above), **all citations valid**,
**citation presence rate = 1.0**.

## What this evaluation does NOT measure

This is the most important section — read it before citing these numbers
as evidence of anything beyond what they actually test.

- **It does not measure real embedding model quality.** The topic-vector
  fake embedder is deliberately exact and orthogonal; it says nothing
  about how `sentence-transformers/all-MiniLM-L6-v2` will rank real,
  messy document text. Retrieval quality on real PDFs can only be
  measured by running the real pipeline against real documents — this
  suite is a regression gate on the *ranking/threshold logic* in
  `retrieval.py`, not a model-quality benchmark.
- **It does not measure whether a real LLM cites well.** The grounded-
  answer evaluation uses a *scripted* client that is programmed to always
  cite correctly — `citation_presence_rate = 1.0` here is a fact about the
  test script, not a claim about Claude, the configured OpenRouter model,
  or any other real model's citation behavior. A real model can still
  fabricate a citation, cite the wrong page, or ignore
  `SYSTEM_PROMPT`'s citation rule; this suite cannot catch that.
- **It is not a large or realistic corpus.** 9 chunks across 2 short
  documents is enough to validate ranking logic and catch regressions, not
  to characterize behavior at the "~5,000 chunks/session" scale
  `retrieval.py`'s own docstring names as its assumed ceiling.
- **A "PASS" from `python -m eval.run_eval` is not a quality score or a
  semantic-accuracy claim.** It means the deterministic pipeline behaves
  as this fixed fixture expects it to, nothing more. Treat a FAIL as a
  real regression signal; treat a PASS as "no regression detected on this
  fixture", not as "retrieval is good."

## Deterministic vs. model-dependent evaluation

Everything described above is deterministic and runs in the standard
offline test suite. A genuine **live-model** evaluation — pointing
`eval.run_eval` at a real OpenRouter-backed model and checking how *it*
phrases citations, handles ambiguous evidence, or recovers from the
repeated-search problem M5's reliability fix targets — is a materially
different, non-deterministic, and (depending on the configured model)
potentially paid exercise. It is **intentionally not implemented** as part
of this suite or wired into `pytest`, per the M5 constraint that ordinary
test runs must never depend on a paid API call or external model
availability. If a live-model evaluation is wanted later, it should be a
clearly separate, explicitly-invoked script (e.g. an `--live` mode) that
reuses `eval/dataset.py`'s questions but swaps `eval/harness.py`'s
scripted client for a real `Anthropic`/OpenRouter client, and reports its
results as a distinct, non-blocking artifact — never folded into the
deterministic numbers above.

## Dataset structure and how to add a case

1. If the question needs new source material, add a chunk to
   `eval/corpus.py`'s `CHUNKS` list: `(filename, page, text,
   topic_vector(...))`. Reuse an existing topic from `TOPICS`, or add a
   new one (adding a topic changes every vector's dimensionality — keep
   `topic_vector()` calls consistent across the whole file).
2. Add a case to `eval/dataset.py`'s `CASES` list:
   ```python
   {
       "id": "short_unique_slug",
       "query": "The exact question text.",
       "query_vector": topic_vector(("topic_name", 1.0)),
       "relevant": [("filename.pdf", page_number), ...],
   }
   ```
3. The query text here must match character-for-character what gets
   passed to `retrieval.search()` / the scripted `search_documents` tool
   call — `eval.harness._fake_embed_texts` looks up vectors by exact
   string and raises loudly (not a silent wrong score) if a case's
   `query` drifts from itself.
4. Run `python -m eval.run_eval` (or `pytest tests/test_eval.py -q`) and
   confirm the new case's `recall_at_k` is `1.0` before committing it.

## Known limitations

- Precision@K and MRR are reported but the suite does not currently gate
  on a precision floor (only recall/citation-validity/presence are
  asserted as regression gates in `tests/test_eval.py`) — a precision
  regression on this fixture would show up in `python -m eval.run_eval`'s
  report but not fail `pytest`.
- The corpus/dataset is hand-authored and small; it is a regression
  fixture, not a benchmark suite, and should grow as real failure modes
  are observed in live usage (the same way the repeated-search problem
  this M5 round fixed was found through live testing, not this suite).
- No live-model evaluation exists yet (see above) — citation-phrasing
  quality against a real model is currently unmeasured by automation.
