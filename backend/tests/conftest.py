"""Shared pytest fixtures for the AEGIS backend test suite.

Historically each test module that touched tracing defined its own
`isolate_trace_file`-style autouse fixture (test_agent.py, test_main.py,
test_documents_endpoints.py, test_skills.py, test_tracing.py). Several
other modules (test_context.py, test_documents.py, test_memory.py,
test_retrieval.py) call app.context.create_session() or other
trace-emitting paths without any such fixture, so a full `pytest` run
appended real session_created/message_appended/etc. records to the real
backend/.traces/traces.jsonl. One autouse fixture here, applied to every
test in the suite regardless of which module defines it or whether that
module remembers to, closes that gap once and for all.
"""
import pytest

from app import tracing


@pytest.fixture(autouse=True)
def isolate_trace_file(tmp_path, monkeypatch):
    """Redirect app.tracing.TRACE_PATH to a per-test tmp file so no test in
    the suite can ever append to the real, on-disk trace file. monkeypatch
    restores the original module attribute automatically at teardown, so
    each test starts from - and the suite as a whole ends on - the real
    TRACE_PATH app.tracing.py defines at import time."""
    monkeypatch.setattr(tracing, "TRACE_PATH", tmp_path / "traces.jsonl")
