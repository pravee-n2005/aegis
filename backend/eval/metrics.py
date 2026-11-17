"""Deterministic retrieval-quality metrics for the M5 evaluation harness.

All functions operate on `(filename, page)` citation tuples, not raw
chunk ids - the question this evaluation answers is "did retrieval surface
the right *pages*", matching what search_documents actually exposes to
the model and what agent.SYSTEM_PROMPT requires it to cite.

Limitations (see docs/EVALUATION.md for the full discussion):
- These numbers describe retrieval behavior on eval/corpus.py's small,
  hand-labeled, deterministically-embedded fixture - not a general
  accuracy claim about real documents or the real embedding model.
- They say nothing about a model's generated *prose*; harness.py's
  run_grounded_answer_eval separately checks citation presence using a
  scripted (not real) LLM client, which is a different, narrower claim
  than "a real model will answer this well".
"""


def recall_at_k(retrieved, relevant, k):
    """Fraction of the relevant set that appears in the top-k retrieved
    results. Requires a non-empty `relevant` set - a case with no ground
    truth can't be scored and is a dataset bug, not a 0.0 result."""
    if not relevant:
        raise ValueError("relevant set must be non-empty")
    top = retrieved[:k]
    hits = len(set(top) & set(relevant))
    return hits / len(relevant)


def precision_at_k(retrieved, relevant, k):
    """Fraction of the top-k retrieved results that are actually relevant.
    0.0 (not an error) when retrieval returned nothing in the top-k - an
    empty result set is a valid, scoreable outcome."""
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = len(set(top) & set(relevant))
    return hits / len(top)


def reciprocal_rank(retrieved, relevant):
    """1/rank of the first relevant result, or 0.0 if none of the
    retrieved results are relevant."""
    relevant_set = set(relevant)
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0
