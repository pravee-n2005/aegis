"""M5: wires the deterministic evaluation harness (eval/harness.py) into
the standard, offline pytest suite as a regression gate on retrieval and
citation quality.

Fully deterministic and offline: the embedding backend is faked with
hand-authored vectors (eval/corpus.py) and the LLM client is scripted
(eval/harness.py's run_grounded_answer_eval), matching the project's
existing test conventions (tests/test_agent.py's fake_embedder_for_retrieval
fixture and _mock_client helper) - no network access, no API key, no real
model. See docs/EVALUATION.md for what these numbers mean and their
limitations; a live-model evaluation is intentionally kept out of this
file and out of the default `pytest` run (see eval/run_eval.py's module
docstring)."""
import os
from pathlib import Path

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

from app import tracing  # noqa: E402
from eval import harness  # noqa: E402
from eval.dataset import CASES  # noqa: E402

# The real, on-disk production trace path app/tracing.py defines at import
# time - computed independently of tracing.TRACE_PATH's current value,
# since conftest.py's autouse isolate_trace_file fixture always has that
# attribute pointed at a per-test tmp file by the time this test body runs.
_REAL_TRACE_PATH = Path(tracing.__file__).resolve().parent.parent / ".traces" / "traces.jsonl"


def test_eval_dataset_every_case_has_a_fixed_query_vector():
    """Sanity check on the dataset itself: harness._fake_embed_texts looks
    vectors up by exact query string, so every case must define one -
    a missing/mismatched entry here would otherwise surface as a confusing
    KeyError deep inside retrieval.search instead of a clear assertion."""
    for case in CASES:
        assert case["query"] in harness._QUERY_VECTORS
        assert case["relevant"], f"case {case['id']!r} must have a non-empty relevant set"


def test_retrieval_eval_achieves_full_recall_on_fixed_corpus(tmp_path):
    """Regression gate: every case's relevant page(s) must be retrievable
    within the default top_k on this fixed, deterministically-embedded
    corpus. A drop below 1.0 here means a real regression in
    retrieval.search's ranking/threshold logic, not noise - the corpus and
    embeddings are fixed, not sampled."""
    report = harness.run_retrieval_eval(tmp_path)
    assert report["mean_recall_at_k"] == 1.0
    assert report["mrr"] == 1.0
    assert report["all_citations_valid"] is True
    for case in report["cases"]:
        assert case["recall_at_k"] == 1.0, f"case {case['id']!r} missed its relevant page(s)"


def test_retrieval_eval_precision_reflects_topic_overlap_not_perfect():
    """Precision@K is NOT expected to be 1.0 for every case: handbook.pdf
    page 5 deliberately blends two topics (security + expense), so a
    single-topic query for either one legitimately also retrieves it.
    This pins that known, intentional behavior - the eval would be a
    non-discriminating no-op if every query scored perfectly on this
    fixture."""
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        report = harness.run_retrieval_eval(Path(tmp))
    by_id = {c["id"]: c for c in report["cases"]}
    assert by_id["security"]["precision_at_k"] < 1.0
    assert by_id["benefits"]["precision_at_k"] == 1.0


def test_grounded_answer_eval_citations_match_real_retrieval(tmp_path):
    """End-to-end pipeline check (real ask_aegis/agent loop, real
    retrieval.search, scripted LLM): the chunk the scripted model cites
    must be the one real retrieval actually surfaced as relevant - proves
    the search_documents -> tool_result -> citation path wired through the
    M5 rounds_remaining/new_chunk_count fields still round-trips real
    citation metadata correctly."""
    report = harness.run_grounded_answer_eval(tmp_path)
    assert report["citation_presence_rate"] == 1.0
    assert report["retrieval_grounding_rate"] == 1.0
    for case in report["cases"]:
        assert case["stop_reason"] == "end_turn"


def test_eval_harness_never_writes_to_the_real_trace_file(tmp_path):
    """Regression gate for the trace-isolation bug: both run_eval.py's CLI
    and this test module drive the real agent loop (ask_aegis) through
    harness.run_grounded_answer_eval, which calls the real write_trace().
    Under pytest, conftest.py's autouse isolate_trace_file fixture has
    already pointed tracing.TRACE_PATH at a per-test tmp file before this
    test body runs - so to prove the harness itself is safe to run
    *outside* pytest too (as `python -m eval.run_eval` is), this checks the
    real, independently-computed production trace path on disk, not
    whatever tracing.TRACE_PATH currently holds. Also asserts
    harness._eval_session restores tracing.TRACE_PATH to whatever value
    was current (here, the conftest-patched tmp path) when it exits."""
    patched_before = tracing.TRACE_PATH
    before = _REAL_TRACE_PATH.read_bytes() if _REAL_TRACE_PATH.exists() else None

    harness.run_grounded_answer_eval(tmp_path)

    assert tracing.TRACE_PATH == patched_before, "TRACE_PATH must be restored after the eval session"
    after = _REAL_TRACE_PATH.read_bytes() if _REAL_TRACE_PATH.exists() else None
    assert after == before, "eval harness must never touch the real trace file"
