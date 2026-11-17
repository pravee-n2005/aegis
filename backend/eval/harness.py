"""Deterministic M5 evaluation harness.

Loads the fixed corpus (corpus.py) into a throwaway SQLite session, answers
the fixed dataset's questions (dataset.py) through the real
retrieval.search(), and (for the grounded-answer check) the real agent
loop ask_aegis() with a scripted LLM client, then reports Recall@K /
Precision@K / MRR plus citation correctness.

Nothing here makes a network or paid-API call - the embedding backend and
the LLM client are both deterministically faked for the duration of a run
(_fake_embed_texts and the scripted Anthropic client below), the same
monkeypatch pattern tests/test_agent.py already uses (its
fake_embedder_for_retrieval fixture and _mock_client helper). This is what
keeps `python -m eval.run_eval` and tests/test_eval.py reproducible and
free to run as often as needed.

Explicitly out of scope here: how a REAL model, given these same tool
results, would choose to phrase or cite its answer. That is a
model-dependent question this deterministic harness cannot measure - see
docs/EVALUATION.md, "What this evaluation does not measure".
"""
import contextlib
import json
import types
import uuid
from unittest.mock import MagicMock, patch

from app import db, embeddings, retrieval, tracing
from app.agent import ask_aegis
from app.config import settings
from app.context import create_session

from . import metrics as m
from .corpus import CHUNKS
from .dataset import CASES

DEFAULT_TOP_K = 3

_QUERY_VECTORS = {case["query"]: case["query_vector"] for case in CASES}
_CORPUS_CITATIONS = {(filename, page) for filename, page, _, _ in CHUNKS}


def _insert_corpus(session_id: str) -> None:
    conn = db.get_connection()
    try:
        document_ids = {}
        for filename, _page, _text, _vector in CHUNKS:
            document_ids.setdefault(filename, uuid.uuid4().hex)
        for filename, document_id in document_ids.items():
            conn.execute(
                "INSERT OR IGNORE INTO documents (id, session_id, filename, page_count, "
                "status, created_at) VALUES (?, ?, ?, 1, 'ready', '2026-01-01T00:00:00+00:00')",
                (document_id, session_id, filename),
            )
        for filename, page, text, vector in CHUNKS:
            conn.execute(
                "INSERT INTO chunks (document_id, session_id, page_number, chunk_index, "
                "content, embedding, created_at) VALUES (?, ?, ?, 0, ?, ?, '2026-01-01T00:00:00+00:00')",
                (document_ids[filename], session_id, page, text, json.dumps(vector)),
            )
        conn.commit()
    finally:
        conn.close()


def _fake_embed_texts(texts):
    """Deterministic stand-in for embeddings.embed_texts: returns the
    hand-authored vector for a known eval query. Raises on anything this
    fixed dataset doesn't define, rather than silently falling back to a
    zero/garbage vector - a case whose query text drifts from its own
    dataset entry (e.g. a copy-paste edit to one but not the other) must
    fail loudly, not quietly score as a retrieval miss."""
    out = []
    for text in texts:
        if text not in _QUERY_VECTORS:
            raise KeyError(f"eval harness has no fixed vector for query: {text!r}")
        out.append(_QUERY_VECTORS[text])
    return out


@contextlib.contextmanager
def _eval_session(tmp_path):
    """Fresh temp-file SQLite DB with the fixed corpus loaded, settings
    pointed at it for the duration of the `with` block, embeddings faked,
    and app.tracing.TRACE_PATH redirected into tmp_path - mirrors
    tests/test_agent.py's isolate_trace_file fixture so the real agent loop
    driven by run_grounded_answer_eval() (ask_aegis) never appends to the
    real backend/.traces/traces.jsonl, whether this runs under pytest or
    the standalone `python -m eval.run_eval` CLI (neither of which gets
    test_agent.py's own fixture, since pytest fixtures are not shared
    across modules without a conftest.py). A plain context manager (not a
    pytest fixture) so both run_eval.py and tests/test_eval.py can use it."""
    original_db_path = settings.db_path
    original_trace_path = tracing.TRACE_PATH
    settings.db_path = str(tmp_path / "aegis_eval.db")
    tracing.TRACE_PATH = tmp_path / "traces.jsonl"
    db.init_db()
    try:
        session_id = create_session()
        _insert_corpus(session_id)
        with patch.object(embeddings, "embed_texts", side_effect=_fake_embed_texts):
            yield session_id
    finally:
        settings.db_path = original_db_path
        tracing.TRACE_PATH = original_trace_path


def run_retrieval_eval(tmp_path, top_k: int = DEFAULT_TOP_K) -> dict:
    """Run every dataset.py case through the real retrieval.search() and
    score it with Recall@K / Precision@K / MRR against the case's
    hand-labeled relevant set. Returns a report dict with per-case and
    averaged metrics, deterministic end to end."""
    per_case = []
    with _eval_session(tmp_path) as session_id:
        for case in CASES:
            results = retrieval.search(session_id, case["query"], top_k=top_k)
            retrieved = [(r["filename"], r["page"]) for r in results]
            relevant = case["relevant"]
            per_case.append({
                "id": case["id"],
                "query": case["query"],
                "relevant": relevant,
                "retrieved": retrieved,
                "recall_at_k": m.recall_at_k(retrieved, relevant, top_k),
                "precision_at_k": m.precision_at_k(retrieved, relevant, top_k),
                "reciprocal_rank": m.reciprocal_rank(retrieved, relevant),
                # Every (filename, page) retrieval returned must actually
                # exist in the corpus under that name - catches a
                # retrieval-layer bug that invents or mismatches a
                # citation, independent of whether the chunk is relevant.
                "citations_exist_in_corpus": all(
                    (r["filename"], r["page"]) in _CORPUS_CITATIONS for r in results
                ),
            })

    return {
        "top_k": top_k,
        "case_count": len(per_case),
        "cases": per_case,
        "mean_recall_at_k": m.mean(c["recall_at_k"] for c in per_case),
        "mean_precision_at_k": m.mean(c["precision_at_k"] for c in per_case),
        "mrr": m.mean(c["reciprocal_rank"] for c in per_case),
        "all_citations_valid": all(c["citations_exist_in_corpus"] for c in per_case),
    }


def _text_block(text):
    return types.SimpleNamespace(type="text", text=text)


def _tool_use_block(name, input_, block_id="t1"):
    return types.SimpleNamespace(type="tool_use", name=name, input=input_, id=block_id)


def _response(content, stop_reason):
    return types.SimpleNamespace(content=content, stop_reason=stop_reason)


def run_grounded_answer_eval(tmp_path) -> dict:
    """For each case, drive the real agent loop (ask_aegis) with a
    scripted LLM client that calls search_documents once and then answers
    citing the top real relevant page. This exercises the full
    search_documents -> tool_result -> citation pipeline end to end
    (including the M5 rounds_remaining/new_chunk_count fields) without a
    real, paid LLM call.

    This measures two different things, and the report keeps them
    separate: whether the real retrieval layer actually surfaced the
    relevant chunk (retrieval_found_relevant_chunk - a real, deterministic
    fact about this codebase), and whether a model that correctly cites
    what it was given would be visible in the final answer text
    (cited_correct_page - true here because the script always cites
    correctly; a REAL model's citation behavior is not what this measures,
    see docs/EVALUATION.md)."""
    per_case = []
    with _eval_session(tmp_path) as session_id:
        for case in CASES:
            expected_filename, expected_page = case["relevant"][0]
            citation = f"({expected_filename}, page {expected_page})"
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = [
                _response([_tool_use_block("search_documents", {"query": case["query"]})], "tool_use"),
                _response([_text_block(f"Per {citation}, here is the answer.")], "end_turn"),
            ]
            with patch("app.agent.Anthropic", return_value=mock_client):
                result = ask_aegis(session_id, case["query"])

            search_result = result["trace"][0]["result"]
            retrieved = [(r["filename"], r["page"]) for r in search_result.get("results", [])]
            per_case.append({
                "id": case["id"],
                "answer": result["answer"],
                "stop_reason": result["stop_reason"],
                "cited_correct_page": citation in result["answer"],
                "retrieval_found_relevant_chunk": (expected_filename, expected_page) in retrieved,
            })

    return {
        "case_count": len(per_case),
        "cases": per_case,
        "citation_presence_rate": m.mean(c["cited_correct_page"] for c in per_case),
        "retrieval_grounding_rate": m.mean(c["retrieval_found_relevant_chunk"] for c in per_case),
    }
