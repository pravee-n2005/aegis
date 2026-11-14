import sqlite3

import pytest

from app import db
from app.config import settings


EXPECTED_TABLES = {"schema_version", "sessions", "messages", "summaries", "memories"}


def _table_names(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {row["name"] for row in rows}


def test_init_db_creates_all_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "aegis.db"))

    db.init_db()

    conn = db.get_connection()
    try:
        assert EXPECTED_TABLES <= _table_names(conn)
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        assert row["version"] == db.SCHEMA_VERSION
    finally:
        conn.close()


def test_get_connection_enables_foreign_keys_and_wal(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "aegis.db"))

    conn = db.get_connection()
    try:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        conn.close()


def test_init_db_permission_denied_raises_oserror(tmp_path, monkeypatch):
    unwritable_parent = tmp_path / "no_such_dir" / "nested"
    monkeypatch.setattr(settings, "db_path", str(unwritable_parent / "aegis.db"))

    def _raise_mkdir(self, parents=True, exist_ok=True):
        raise OSError("permission denied")

    monkeypatch.setattr("pathlib.Path.mkdir", _raise_mkdir)

    with pytest.raises(OSError):
        db.init_db()


def test_init_db_is_idempotent_on_existing_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "aegis.db"))

    db.init_db()
    db.init_db()

    conn = db.get_connection()
    try:
        assert EXPECTED_TABLES <= _table_names(conn)
        rows = conn.execute("SELECT COUNT(*) AS c FROM schema_version").fetchone()
        assert rows["c"] == 1
    finally:
        conn.close()


def test_messages_cascade_deletes_with_session(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "aegis.db"))
    db.init_db()

    conn = db.get_connection()
    try:
        conn.execute(
            "INSERT INTO sessions (id, created_at, last_active_at) VALUES ('s1', 'now', 'now')"
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, token_estimate, created_at) "
            "VALUES ('s1', 'user', '{}', 1, 'now')"
        )
        conn.commit()

        conn.execute("DELETE FROM sessions WHERE id = 's1'")
        conn.commit()

        remaining = conn.execute("SELECT COUNT(*) AS c FROM messages WHERE session_id = 's1'").fetchone()
        assert remaining["c"] == 0
    finally:
        conn.close()
