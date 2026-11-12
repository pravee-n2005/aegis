"""Append-only JSONL execution tracing for the agent loop.

Fixed path, no configurable backend (YAGNI until a second consumer needs one).
A trace write failure must never break a chat response, so every write is
best-effort and swallows its own errors.
"""
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

TRACE_PATH = Path(__file__).resolve().parent.parent / ".traces" / "traces.jsonl"
MAX_FIELD_CHARS = 200


def new_request_id() -> str:
    return uuid.uuid4().hex


def redact(value) -> str:
    """Truncate a value to a bounded string. This does NOT detect or scrub
    secrets — it only limits length. Never pass raw user messages, tool
    input or tool output through here for persistence: callers that might
    carry user-controlled or workspace content must log length/metadata
    only, not content, however this function truncates it."""
    text = json.dumps(value, default=str) if not isinstance(value, str) else value
    return text if len(text) <= MAX_FIELD_CHARS else text[:MAX_FIELD_CHARS] + "...(truncated)"


def content_length(value) -> int:
    """Length of a value's JSON-serialized form, for logging size without content."""
    try:
        return len(value) if isinstance(value, str) else len(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return -1


def write_trace(event: str, request_id: str, **fields) -> None:
    record = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    try:
        TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRACE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        print(f"aegis-trace: write failed: {exc}", file=sys.stderr)
