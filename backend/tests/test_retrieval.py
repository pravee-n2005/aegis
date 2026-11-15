import json

import pytest

from app import context, db, embeddings, retrieval
from app.config import settings


@pytest.fixture(autouse=True)
def isolate_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "aegis.db"))
    db.init_db()


_VECTORS = {
    "query": [1.0, 0.0, 0.0],
    "about apples": [0.9, 0.1, 0.0],
    "about bananas": [0.0, 1.0, 0.0],
}


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    def _fake_embed_texts(texts):
        return [_VECTORS.get(t, [0.0, 0.0, 1.0]) for t in texts]
    monkeypatch.setattr(embeddings, "embed_texts", _fake_embed_texts)


def _insert_document(session_id, document_id, filename="doc.pdf", page_count=1):
    conn = db.get_connection()
    try:
        conn.execute(
            "INSERT INTO documents (id, session_id, filename, page_count, status, created_at) "
            "VALUES (?, ?, ?, ?, 'ready', '2026-01-01T00:00:00+00:00')",
            (document_id, session_id, filename, page_count),
        )
        conn.commit()
    finally:
        conn.close()


def _insert_chunk(session_id, document_id, page_number, content, embedding_json):
    conn = db.get_connection()
    try:
        conn.execute(
            "INSERT INTO chunks (document_id, session_id, page_number, chunk_index, content, "
            "embedding, created_at) VALUES (?, ?, ?, 0, ?, ?, '2026-01-01T00:00:00+00:00')",
            (document_id, session_id, page_number, content, embedding_json),
        )
        conn.commit()
    finally:
        conn.close()


def test_search_ranks_by_cosine_similarity_highest_first():
    session_id = context.create_session()
    _insert_document(session_id, "doc-1", filename="fruit.pdf")
    _insert_chunk(session_id, "doc-1", 1, "about bananas", json.dumps(_VECTORS["about bananas"]))
    _insert_chunk(session_id, "doc-1", 2, "about apples", json.dumps(_VECTORS["about apples"]))

    results = retrieval.search(session_id, "query", top_k=5)

    assert [r["text"] for r in results] == ["about apples", "about bananas"]
    assert results[0]["filename"] == "fruit.pdf"
    assert results[0]["page"] == 2
    assert results[0]["score"] > results[1]["score"]


def test_search_respects_top_k():
    session_id = context.create_session()
    _insert_document(session_id, "doc-1")
    for i in range(10):
        _insert_chunk(session_id, "doc-1", i + 1, f"chunk {i}", json.dumps([0.0, 0.0, 1.0]))

    results = retrieval.search(session_id, "query", top_k=3)

    assert len(results) == 3


def test_search_no_chunks_returns_empty_list():
    session_id = context.create_session()
    assert retrieval.search(session_id, "query") == []


def test_search_cross_session_isolation():
    session_a = context.create_session()
    session_b = context.create_session()
    _insert_document(session_a, "doc-a")
    _insert_chunk(session_a, "doc-a", 1, "about apples", json.dumps(_VECTORS["about apples"]))

    assert retrieval.search(session_b, "query") == []
    assert len(retrieval.search(session_a, "query")) == 1


def test_search_skips_corrupted_embedding_row_without_crashing():
    session_id = context.create_session()
    _insert_document(session_id, "doc-1")
    _insert_chunk(session_id, "doc-1", 1, "corrupted row", "not valid json")
    _insert_chunk(session_id, "doc-1", 2, "about apples", json.dumps(_VECTORS["about apples"]))

    results = retrieval.search(session_id, "query")

    assert len(results) == 1
    assert results[0]["text"] == "about apples"


def test_search_skips_shape_mismatched_embedding_without_wrong_score():
    """A JSON-valid but wrong-shaped/typed stored embedding (short vector, a
    scalar, an object) must be skipped like a parse failure - never silently
    scored via a zip()-truncated dot product, which would rank it on a
    meaningless partial comparison instead of rejecting it."""
    session_id = context.create_session()
    _insert_document(session_id, "doc-1")
    _insert_chunk(session_id, "doc-1", 1, "short vector", json.dumps([1.0, 0.0]))
    _insert_chunk(session_id, "doc-1", 2, "not a list", json.dumps("5"))
    _insert_chunk(session_id, "doc-1", 3, "about apples", json.dumps(_VECTORS["about apples"]))

    results = retrieval.search(session_id, "query")

    assert len(results) == 1
    assert results[0]["text"] == "about apples"
