import os
import types
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

from app import context, db, documents, embeddings  # noqa: E402
from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_trace_and_db(tmp_path, monkeypatch):
    """TRACE_PATH isolation now comes from conftest.py's autouse
    isolate_trace_file fixture; this fixture keeps its skills_dir/db_path
    isolation."""
    monkeypatch.setattr(settings, "skills_dir", str(tmp_path / "skills"))
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "aegis.db"))
    db.init_db()


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    """No endpoint test downloads or runs the real sentence-transformers model."""
    def _fake_embed_texts(texts):
        return [[float(len(t)), 0.0, 0.0] for t in texts]
    monkeypatch.setattr(embeddings, "embed_texts", _fake_embed_texts)


def _make_pdf(text: str, pages: int = 1) -> bytes:
    """Build a minimal, valid, hand-written PDF with real extractable text
    (no reportlab/third-party dependency needed for a test fixture) - same
    approach as test_documents.py's fixture builder."""
    objects = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{3 + i} 0 R" for i in range(pages))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode())
    page_obj_start = 3
    font_obj = page_obj_start + pages
    content_obj_start = font_obj + 1
    for i in range(pages):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 {font_obj} 0 R >> >> "
            f"/MediaBox [0 0 612 792] /Contents {content_obj_start + i} 0 R >>".encode()
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i in range(pages):
        stream = f"BT /F1 24 Tf 72 712 Td ({text} page {i + 1}) Tj ET".encode()
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")

    out = [b"%PDF-1.4\n"]
    offsets = [0]
    pos = len(out[0])
    for idx, obj in enumerate(objects, start=1):
        entry = f"{idx} 0 obj\n".encode() + obj + b"\nendobj\n"
        offsets.append(pos)
        out.append(entry)
        pos += len(entry)
    xref_offset = pos
    n = len(objects) + 1
    xref = [f"xref\n0 {n}\n".encode(), b"0000000000 65535 f \n"]
    for off in offsets[1:]:
        xref.append(f"{off:010d} 00000 n \n".encode())
    out.extend(xref)
    out.append(f"trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode())
    return b"".join(out)


def _make_blank_pdf() -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
    ]
    out = [b"%PDF-1.4\n"]
    offsets = [0]
    pos = len(out[0])
    for idx, obj in enumerate(objects, start=1):
        entry = f"{idx} 0 obj\n".encode() + obj + b"\nendobj\n"
        offsets.append(pos)
        out.append(entry)
        pos += len(entry)
    xref_offset = pos
    n = len(objects) + 1
    xref = [f"xref\n0 {n}\n".encode(), b"0000000000 65535 f \n"]
    for off in offsets[1:]:
        xref.append(f"{off:010d} 00000 n \n".encode())
    out.extend(xref)
    out.append(f"trailer\n<< /Size {n} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode())
    return b"".join(out)


def _session():
    return context.create_session()


def test_upload_document_happy_path():
    session_id = _session()
    pdf_bytes = _make_pdf("Hello RAG world", pages=2)

    response = client.post(f"/api/sessions/{session_id}/documents",
                            files={"file": ("sample.pdf", pdf_bytes, "application/pdf")})

    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "sample.pdf"
    assert body["page_count"] == 2
    assert body["status"] == "ready"
    assert "document_id" in body


def test_upload_document_empty_scanned_pdf_returns_empty_status_zero_chunks():
    session_id = _session()
    response = client.post(f"/api/sessions/{session_id}/documents",
                            files={"file": ("scanned.pdf", _make_blank_pdf(), "application/pdf")})

    assert response.status_code == 200
    assert response.json()["status"] == "empty"
    assert documents.list_chunks(session_id) == []


def test_upload_document_unknown_session_returns_404():
    response = client.post("/api/sessions/does-not-exist/documents",
                            files={"file": ("sample.pdf", _make_pdf("hi"), "application/pdf")})
    assert response.status_code == 404


def test_upload_document_malformed_pdf_returns_422_no_partial_rows():
    session_id = _session()
    response = client.post(f"/api/sessions/{session_id}/documents",
                            files={"file": ("bad.pdf", b"not a pdf at all", "application/pdf")})

    assert response.status_code == 422
    assert documents.list_documents(session_id) == []


def test_upload_document_oversized_returns_413_before_extraction(monkeypatch):
    session_id = _session()
    monkeypatch.setattr(documents, "MAX_UPLOAD_BYTES", 10)

    response = client.post(f"/api/sessions/{session_id}/documents",
                            files={"file": ("big.pdf", b"x" * 1000, "application/pdf")})

    assert response.status_code == 413
    assert documents.list_documents(session_id) == []


def test_upload_document_page_limit_exceeded_returns_422(monkeypatch):
    session_id = _session()
    monkeypatch.setattr(documents, "MAX_PAGES", 1)

    response = client.post(f"/api/sessions/{session_id}/documents",
                            files={"file": ("toolong.pdf", _make_pdf("overflow", pages=2), "application/pdf")})

    assert response.status_code == 422
    assert documents.list_documents(session_id) == []


def test_upload_document_embedding_unavailable_returns_503_no_partial_rows(monkeypatch):
    session_id = _session()

    def _raise(_texts):
        raise embeddings.EmbeddingUnavailableError("model failed to load")
    monkeypatch.setattr(embeddings, "embed_texts", _raise)

    response = client.post(f"/api/sessions/{session_id}/documents",
                            files={"file": ("sample.pdf", _make_pdf("hello"), "application/pdf")})

    assert response.status_code == 503
    assert documents.list_documents(session_id) == []


def test_upload_document_chunk_limit_exceeded_returns_422(monkeypatch):
    session_id = _session()
    monkeypatch.setattr(documents, "MAX_CHUNKS_PER_DOCUMENT", 1)

    response = client.post(f"/api/sessions/{session_id}/documents",
                            files={"file": ("dense.pdf", _make_pdf("a " * 2000, pages=2), "application/pdf")})

    assert response.status_code == 422
    assert documents.list_documents(session_id) == []


def test_list_documents_returns_uploaded_documents():
    session_id = _session()
    client.post(f"/api/sessions/{session_id}/documents",
                files={"file": ("a.pdf", _make_pdf("a"), "application/pdf")})
    client.post(f"/api/sessions/{session_id}/documents",
                files={"file": ("b.pdf", _make_pdf("b"), "application/pdf")})

    response = client.get(f"/api/sessions/{session_id}/documents")

    assert response.status_code == 200
    filenames = {d["filename"] for d in response.json()["documents"]}
    assert filenames == {"a.pdf", "b.pdf"}


def test_list_documents_unknown_session_returns_404():
    response = client.get("/api/sessions/does-not-exist/documents")
    assert response.status_code == 404


def test_delete_document_removes_it():
    session_id = _session()
    upload = client.post(f"/api/sessions/{session_id}/documents",
                          files={"file": ("sample.pdf", _make_pdf("hello"), "application/pdf")})
    document_id = upload.json()["document_id"]

    response = client.delete(f"/api/sessions/{session_id}/documents/{document_id}")

    assert response.status_code == 200
    assert documents.list_documents(session_id) == []


def test_delete_document_unknown_id_returns_404():
    session_id = _session()
    response = client.delete(f"/api/sessions/{session_id}/documents/does-not-exist")
    assert response.status_code == 404


def test_upload_then_chat_question_gets_insufficient_evidence_style_answer():
    """Phase 3 acceptance: an empty/scanned-PDF upload returning status
    "empty" with zero chunks, then a subsequent chat question getting an
    "insufficient evidence" style answer from a mocked response."""
    session_id = _session()
    client.post(f"/api/sessions/{session_id}/documents",
                files={"file": ("scanned.pdf", _make_blank_pdf(), "application/pdf")})

    def _text_block(text):
        return types.SimpleNamespace(type="text", text=text,
                                      model_dump=lambda: {"type": "text", "text": text})

    def _tool_use_block(name, input_, block_id="tool_1"):
        return types.SimpleNamespace(type="tool_use", name=name, input=input_, id=block_id,
                                      model_dump=lambda: {"type": "tool_use", "name": name,
                                                           "input": input_, "id": block_id})

    def _response(content, stop_reason):
        return types.SimpleNamespace(content=content, stop_reason=stop_reason)

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        _response([_tool_use_block("search_documents", {"query": "what does the document say"})], "tool_use"),
        _response([_text_block("Insufficient evidence to answer that from the uploaded document.")], "end_turn"),
    ]
    with patch("app.agent.Anthropic", return_value=mock_client):
        response = client.post("/api/chat", json={"message": "What does my document say?",
                                                    "session_id": session_id})

    assert response.status_code == 200
    body = response.json()
    assert "insufficient evidence" in body["answer"].lower()
    assert body["trace"][0]["tool"] == "search_documents"
    assert body["trace"][0]["result"]["results"] == []


# --- M5: cross-session document isolation at the API layer (gap fill, 2026-11-16) ---
#
# documents.py / retrieval.py already have unit-level cross-session
# isolation tests (test_documents.py, test_retrieval.py). These exercise
# the same guarantees through the real HTTP endpoints and the real chat
# pipeline end to end, since an endpoint-level or chat-pipeline-level
# regression (e.g. a route reading the wrong session_id) would not be
# caught by a unit test that calls documents.py/retrieval.py directly.

def test_delete_document_from_different_session_returns_404_not_success():
    """Session B must not be able to delete session A's document by id,
    even though document ids are globally unique and B knows a valid one -
    the delete must be scoped by (document_id AND session_id)."""
    owner_session = _session()
    attacker_session = _session()
    upload = client.post(f"/api/sessions/{owner_session}/documents",
                          files={"file": ("mine.pdf", _make_pdf("secret"), "application/pdf")})
    document_id = upload.json()["document_id"]

    response = client.delete(f"/api/sessions/{attacker_session}/documents/{document_id}")

    assert response.status_code == 404
    assert documents.list_documents(owner_session)[0]["document_id"] == document_id


def test_chat_in_one_session_cannot_retrieve_another_sessions_document_content():
    """End-to-end cross-session RAG isolation through the real chat
    endpoint: session B asks a question and the real search_documents tool
    call must return nothing from session A's uploaded document, even
    though both sessions share the same process and database file."""
    session_a = _session()
    session_b = _session()
    client.post(f"/api/sessions/{session_a}/documents",
                files={"file": ("confidential.pdf", _make_pdf("top secret payroll data"), "application/pdf")})

    def _text_block(text):
        return types.SimpleNamespace(type="text", text=text,
                                      model_dump=lambda: {"type": "text", "text": text})

    def _tool_use_block(name, input_, block_id="tool_1"):
        return types.SimpleNamespace(type="tool_use", name=name, input=input_, id=block_id,
                                      model_dump=lambda: {"type": "tool_use", "name": name,
                                                           "input": input_, "id": block_id})

    def _response(content, stop_reason):
        return types.SimpleNamespace(content=content, stop_reason=stop_reason)

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        _response([_tool_use_block("search_documents", {"query": "payroll data"})], "tool_use"),
        _response([_text_block("I have no document content to answer that from.")], "end_turn"),
    ]
    with patch("app.agent.Anthropic", return_value=mock_client):
        response = client.post("/api/chat", json={"message": "What does the payroll document say?",
                                                    "session_id": session_b})

    assert response.status_code == 200
    body = response.json()
    assert body["trace"][0]["result"]["results"] == []
