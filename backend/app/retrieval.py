"""Brute-force cosine-similarity retrieval over a session's document chunks.

Pure ranking logic over rows handed to it - owns no SQL against `chunks`.
Always goes through documents.list_chunks(session_id), matching db.py's
"one owning module per table" convention (memory.py's docstring: "build_context
calls memory.list_memories() rather than querying the memories table directly,
so no second module ever learns that schema").

Assumed adequate up to ~5,000 chunks/session (docs/designs/
m4-rag-document-intelligence.md, "Dependency choices and trade-offs") - the
`retrieval` trace event's chunk_count field lets this ceiling be checked
against real usage rather than left an unverified guess.
"""
import json
import math

from . import documents
from . import embeddings
from .config import settings
from .tracing import new_request_id, redact, write_trace

EVENT_RETRIEVAL = "retrieval"
EVENT_RETRIEVAL_SKIPPED = "retrieval_row_skipped"

DEFAULT_TOP_K = 5


def _cosine(a: list[float], b: list[float]) -> float:
    if (not isinstance(b, list) or len(b) != len(a)
            or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in b)):
        # A JSON-valid but wrong-shaped/typed/non-finite stored embedding (an
        # object, a scalar, a short/long vector, or NaN/Infinity - Python's
        # json module accepts those non-standard literals by default) must be
        # rejected here, not silently zip()-truncated into a wrong-but-
        # plausible score, or left to corrupt the sort with a NaN a few lines
        # down - caller treats ValueError the same as a JSON parse failure
        # (CORRUPTED_EMBEDDING_SKIPPED).
        raise ValueError(f"stored embedding has shape/type {type(b).__name__}, "
                          f"expected a {len(a)}-element finite numeric list")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def search(session_id: str, query: str, top_k: int = DEFAULT_TOP_K,
           request_id: str = None, min_score: float = None) -> list[dict]:
    """Embed `query`, rank this session's chunks by cosine similarity, and
    return the top_k as {"filename", "page", "text", "score"} dicts, highest
    score first. A row whose stored embedding fails to parse is skipped (a
    CORRUPTED_EMBEDDING_SKIPPED trace event is emitted) rather than crashing
    the whole search - matching context.py's CORRUPTED_SUMMARY_SKIPPED
    precedent for a truncated/corrupted historical write.

    A chunk scoring below `min_score` (default: settings.search_relevance_threshold,
    a measured value - see config.py) is never returned, however few chunks
    the session has: an irrelevant top-K match is not supporting evidence,
    it's noise that invites a confidently-cited wrong answer. Returns []
    when nothing meets the threshold, same shape as "no chunks at all"."""
    rid = request_id or new_request_id()
    threshold = settings.search_relevance_threshold if min_score is None else min_score
    rows = documents.list_chunks(session_id)
    if not rows:
        return []

    [query_vector] = embeddings.embed_texts([query])

    scored = []
    below_threshold = 0
    for row in rows:
        try:
            embedding = json.loads(row["embedding"])
            score = _cosine(query_vector, embedding)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            write_trace(EVENT_RETRIEVAL_SKIPPED, rid, reason="CORRUPTED_EMBEDDING_SKIPPED",
                        session_id=session_id, chunk_id=row["id"], error=redact(str(exc)))
            continue
        if score < threshold:
            below_threshold += 1
            continue
        scored.append((score, row))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = scored[:top_k]

    write_trace(EVENT_RETRIEVAL, rid, session_id=session_id, chunk_count=len(rows),
                scored_count=len(scored) + below_threshold, returned_count=len(top),
                below_threshold_count=below_threshold, threshold=threshold)

    return [
        {"chunk_id": row["id"], "filename": row["filename"], "page": row["page_number"],
         "text": row["content"], "score": score}
        for score, row in top
    ]
