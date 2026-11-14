"""SQLite connection and schema management for AEGIS session persistence.

Single connection-creation path (T10, docs/designs/m3-context-memory.md):
every connection in this codebase must go through get_connection() so that
PRAGMA foreign_keys=ON is always set — SQLite scopes that pragma per
connection, not per database file, so a raw sqlite3.connect() call anywhere
else would silently stop cascading deletes while the parent row still
disappears.

Schema: sessions / messages / summaries / memories, each scoped by
session_id with an index covering that filter, plus a single-row
schema_version table so a future migration has a version to check against
(not used by M3 itself - first schema version, nothing to migrate from).
"""
import sqlite3
from pathlib import Path

from .config import settings

SCHEMA_VERSION = 1


def get_connection() -> sqlite3.Connection:
    """Open a new connection to the AEGIS SQLite database. Always sets
    foreign_keys=ON and WAL mode (concurrent reads during a write, so GET
    endpoints don't block on POST /api/chat). Raises sqlite3.OperationalError
    or OSError on failure (locked file, disk full, permission denied) -
    callers map these to a 503, never let them propagate as a bare 500."""
    db_path = Path(settings.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db() -> None:
    """Create the schema if it doesn't already exist. Uses CREATE TABLE IF
    NOT EXISTS throughout, so it's safe to call on every startup against a
    fresh or already-migrated database file."""
    conn = get_connection()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                last_active_at TEXT NOT NULL,
                skill_allowed_tools TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                token_estimate INTEGER NOT NULL,
                summarized INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
            CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id, id);
            CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id, id);
            """
        )
        row = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()
        if row[0] == 0:
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        conn.commit()
    finally:
        conn.close()
