"""Session-scoped long-term memory: heuristic extraction and CRUD.

Extraction is a deterministic regex over explicit user statements
("remember that X"), not an LLM call - required by the "no unnecessary live
LLM calls" constraint. This has real false positives/negatives by design;
delete_memory is the correction mechanism, not a classifier improvement
(docs/designs/m3-context-memory.md).
"""
import re
import sqlite3
from datetime import datetime, timezone

from . import db
from .tracing import new_request_id, redact, write_trace

EVENT_EXTRACTED = "memory_extracted"
EVENT_DELETED = "memory_deleted"

_REMEMBER_RE = re.compile(r"remember (?:that )?(.+)", re.IGNORECASE)


class MemoryNotFoundError(Exception):
    def __init__(self, memory_id):
        super().__init__(f"Unknown memory_id: {memory_id!r}")
        self.memory_id = memory_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_memories(session_id: str, message: str, request_id: str = None) -> list:
    """Heuristically extract and persist any explicit "remember ..."
    statement in message. Never raises - a malformed/unmatchable message
    just means no new memory this turn."""
    rid = request_id or new_request_id()
    try:
        match = _REMEMBER_RE.search(message)
        content = match.group(1).strip().rstrip(".") if match else ""
    except (re.error, ValueError) as exc:
        write_trace(EVENT_EXTRACTED, rid, reason="ERROR", session_id=session_id, error=redact(str(exc)))
        return []

    if not content:
        write_trace(EVENT_EXTRACTED, rid, reason="NO_MATCH", session_id=session_id)
        return []

    now = _now()
    conn = db.get_connection()
    try:
        conn.execute(
            "INSERT INTO memories (session_id, content, created_at) VALUES (?, ?, ?)",
            (session_id, content, now),
        )
        conn.commit()
    except sqlite3.IntegrityError as exc:
        write_trace(EVENT_EXTRACTED, rid, reason="UNKNOWN_SESSION", session_id=session_id, error=redact(str(exc)))
        return []
    finally:
        conn.close()

    write_trace(EVENT_EXTRACTED, rid, reason="OK", session_id=session_id, count=1)
    return [content]


def list_memories(session_id: str) -> list:
    conn = db.get_connection()
    try:
        rows = conn.execute(
            "SELECT id, content, created_at FROM memories WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()
    return [{"id": r["id"], "content": r["content"], "created_at": r["created_at"]} for r in rows]


def delete_memory(session_id: str, memory_id: int, request_id: str = None) -> None:
    conn = db.get_connection()
    try:
        cursor = conn.execute(
            "DELETE FROM memories WHERE id = ? AND session_id = ?", (memory_id, session_id)
        )
        conn.commit()
    finally:
        conn.close()
    if cursor.rowcount == 0:
        raise MemoryNotFoundError(memory_id)
    write_trace(EVENT_DELETED, request_id or new_request_id(), session_id=session_id, memory_id=memory_id)


def clear_memories(session_id: str, request_id: str = None) -> int:
    """Delete every memory for a session without deleting the session
    itself. Separate from the cascade-delete that DELETE /api/sessions/{id}
    already gets for free via the foreign key - this is the explicit
    "forget everything, keep chatting" operation."""
    conn = db.get_connection()
    try:
        cursor = conn.execute("DELETE FROM memories WHERE session_id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()
    write_trace(EVENT_DELETED, request_id or new_request_id(), session_id=session_id, count=cursor.rowcount)
    return cursor.rowcount
