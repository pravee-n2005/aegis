import json
from pathlib import Path

import pytest

from app import tracing


@pytest.fixture(autouse=True)
def isolate_trace_file(tmp_path, monkeypatch):
    monkeypatch.setattr(tracing, "TRACE_PATH", tmp_path / "traces.jsonl")


def test_write_trace_appends_jsonl_record():
    tracing.write_trace("tool_call", "req-1", tool="calculator", ok=True, input_length=5)
    lines = tracing.TRACE_PATH.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "tool_call"
    assert record["request_id"] == "req-1"
    assert record["tool"] == "calculator"
    assert record["input_length"] == 5
    assert "timestamp" in record


def test_write_trace_appends_multiple_records_in_order():
    tracing.write_trace("tool_call", "req-1", tool="calculator")
    tracing.write_trace("final", "req-1", stop_reason="end_turn")
    lines = tracing.TRACE_PATH.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "tool_call"
    assert json.loads(lines[1])["event"] == "final"


def test_write_trace_swallows_exceptions_and_never_raises(monkeypatch, capsys):
    def _raise_mkdir(self, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "mkdir", _raise_mkdir)
    tracing.write_trace("final", "req-1")  # must not raise
    assert "write failed" in capsys.readouterr().err


def test_write_trace_swallows_non_oserror_exceptions(monkeypatch, capsys):
    def _raise_type_error(*args, **kwargs):
        raise TypeError("not serializable")

    monkeypatch.setattr(tracing.json, "dumps", _raise_type_error)
    tracing.write_trace("final", "req-1")  # must not raise even on a non-OSError failure
    assert "write failed" in capsys.readouterr().err


def test_redact_truncates_long_values():
    long_value = "x" * (tracing.MAX_FIELD_CHARS + 50)
    redacted = tracing.redact(long_value)
    assert redacted.endswith("...(truncated)")
    assert len(redacted) <= tracing.MAX_FIELD_CHARS + len("...(truncated)")


def test_redact_passes_through_short_values():
    assert tracing.redact("short") == "short"


def test_content_length_for_dict():
    assert tracing.content_length({"expression": "2+2"}) == len(json.dumps({"expression": "2+2"}))


def test_content_length_for_string():
    assert tracing.content_length("hello") == 5
