import io
import json
import time

import pytest

from app import context, db, documents, embeddings
from app.config import settings


# Module-level (picklable) fake extraction workers, injected via
# documents._extract_pdf_isolated's/documents.ingest()'s worker lookup to
# test the isolation boundary deterministically - the "spawn" context pickles
# a worker by module+name reference, so it cannot be a local closure/lambda.

def _worker_reports_corrupted(file_bytes, result_queue, max_pages, memory_bytes, cpu_seconds):
    result_queue.put(("corrupted", "simulated pypdf internal failure, not a PdfReadError subclass"))


def _worker_hangs_forever(file_bytes, result_queue, max_pages, memory_bytes, cpu_seconds):
    time.sleep(3600)


def _worker_exits_abnormally(file_bytes, result_queue, max_pages, memory_bytes, cpu_seconds):
    import os
    os._exit(1)


def _worker_reports_ok(file_bytes, result_queue, max_pages, memory_bytes, cpu_seconds):
    result_queue.put(("ok", 1, ["real extracted text"]))


@pytest.fixture(autouse=True)
def isolate_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "aegis.db"))
    db.init_db()


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    """No test downloads or runs the real sentence-transformers model."""
    def _fake_embed_texts(texts):
        return [[float(len(t)), 0.0, 0.0] for t in texts]
    monkeypatch.setattr(embeddings, "embed_texts", _fake_embed_texts)


def _make_pdf(text: str, pages: int = 1) -> bytes:
    """Build a minimal, valid, hand-written PDF with real extractable text
    (no reportlab/third-party dependency needed for a test fixture)."""
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
    """A structurally valid PDF with a page that has no text content at
    all - the "scanned/no extractable text" case."""
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


def test_ingest_valid_pdf_produces_expected_rows_and_pages():
    session_id = context.create_session()
    pdf_bytes = _make_pdf("Hello RAG world", pages=2)

    result = documents.ingest(session_id, "sample.pdf", pdf_bytes)

    assert result["page_count"] == 2
    assert result["status"] == "ready"
    chunks = documents.list_chunks(session_id)
    assert len(chunks) == 2
    assert {c["page_number"] for c in chunks} == {1, 2}
    assert "Hello RAG world page 1" in chunks[0]["content"] or "Hello RAG world page 1" in chunks[1]["content"]
    for c in chunks:
        assert json.loads(c["embedding"]) == [float(len(c["content"])), 0.0, 0.0]


def test_ingest_blank_pdf_is_empty_status_not_error():
    session_id = context.create_session()
    pdf_bytes = _make_blank_pdf()

    result = documents.ingest(session_id, "scanned.pdf", pdf_bytes)

    assert result["status"] == "empty"
    assert documents.list_chunks(session_id) == []
    [doc] = documents.list_documents(session_id)
    assert doc["status"] == "empty"


def test_ingest_oversized_upload_raises_before_parsing(monkeypatch):
    session_id = context.create_session()
    monkeypatch.setattr(documents, "MAX_UPLOAD_BYTES", 10)

    with pytest.raises(documents.UploadTooLargeError):
        documents.ingest(session_id, "big.pdf", b"x" * 100)

    assert documents.list_documents(session_id) == []


def test_ingest_malformed_pdf_raises_corrupted_no_partial_rows():
    session_id = context.create_session()

    with pytest.raises(documents.CorruptedPDFError):
        documents.ingest(session_id, "not-a-pdf.pdf", b"this is not a pdf file at all")

    assert documents.list_documents(session_id) == []
    assert documents.list_chunks(session_id) == []


def test_ingest_page_limit_exceeded_raises_no_partial_rows(monkeypatch):
    session_id = context.create_session()
    monkeypatch.setattr(documents, "MAX_PAGES", 1)
    pdf_bytes = _make_pdf("overflow", pages=2)

    with pytest.raises(documents.PageLimitExceededError):
        documents.ingest(session_id, "toolong.pdf", pdf_bytes)

    assert documents.list_documents(session_id) == []


def test_ingest_chunk_limit_exceeded_raises_no_partial_rows(monkeypatch):
    session_id = context.create_session()
    monkeypatch.setattr(documents, "MAX_CHUNKS_PER_DOCUMENT", 1)
    pdf_bytes = _make_pdf("a " * 2000, pages=2)

    with pytest.raises(documents.ChunkLimitExceededError):
        documents.ingest(session_id, "dense.pdf", pdf_bytes)

    assert documents.list_documents(session_id) == []
    assert documents.list_chunks(session_id) == []


def test_ingest_single_page_text_bomb_raises_before_chunking(monkeypatch):
    """A single pathological page's extracted text, not spread across many
    chunks, must be rejected on its own - MAX_CHUNKS_PER_DOCUMENT alone
    can't catch this since it only trips after chunking already happened."""
    session_id = context.create_session()
    monkeypatch.setattr(documents, "MAX_PAGE_TEXT_CHARS", 10)
    pdf_bytes = _make_pdf("more than ten characters of text", pages=1)

    with pytest.raises(documents.ChunkLimitExceededError):
        documents.ingest(session_id, "bomb.pdf", pdf_bytes)

    assert documents.list_documents(session_id) == []
    assert documents.list_chunks(session_id) == []


def test_ingest_extraction_timeout_kills_worker_raises_no_partial_rows(monkeypatch):
    """A hung extraction worker (e.g. a decompression bomb spinning the CPU,
    or stuck in a C-level loop no Python-level cap can interrupt) must be
    killed by the parent's wall-clock timeout, not left running - this is
    the actual resource boundary, not the post-extraction character cap."""
    session_id = context.create_session()
    monkeypatch.setattr(documents, "PDF_EXTRACTION_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(documents, "_extraction_worker", _worker_hangs_forever)

    with pytest.raises(documents.PDFExtractionError, match="time limit"):
        documents.ingest(session_id, "hangs.pdf", _make_pdf("hello"))

    assert documents.list_documents(session_id) == []
    assert documents.list_chunks(session_id) == []


def test_ingest_extraction_abnormal_exit_raises_no_partial_rows(monkeypatch):
    """A worker process that dies outright (OOM-killed by the OS for
    exceeding its memory limit, a hard crash, a signal) must be detected via
    its exit code and reported as PDFExtractionError, not silently treated
    as success or hang the parent waiting on an empty queue."""
    session_id = context.create_session()
    monkeypatch.setattr(documents, "_extraction_worker", _worker_exits_abnormally)

    with pytest.raises(documents.PDFExtractionError, match="abnormally"):
        documents.ingest(session_id, "crashes.pdf", _make_pdf("hello"))

    assert documents.list_documents(session_id) == []
    assert documents.list_chunks(session_id) == []


def test_extract_pdf_isolated_happy_path_via_injected_worker():
    """Sanity check that the isolation plumbing itself (spawn a process,
    read its queue, return (page_count, page_texts)) works end to end for
    a worker that reports success, independent of real PDF parsing."""
    page_count, page_texts = documents._extract_pdf_isolated(
        b"irrelevant for this fake worker", worker=_worker_reports_ok)

    assert page_count == 1
    assert page_texts == ["real extracted text"]


def test_ingest_embedder_vector_count_mismatch_raises_no_partial_rows(monkeypatch):
    """embed_texts() must return exactly one vector per input chunk; a
    backend that returns fewer must not silently commit a document marked
    "ready" with a chunk_count that doesn't match what was actually
    inserted (zip()'s silent-truncation failure mode)."""
    session_id = context.create_session()

    def _short_embed_texts(texts):
        return [[0.0, 0.0, 0.0] for _ in texts[:-1]] if texts else []
    monkeypatch.setattr(embeddings, "embed_texts", _short_embed_texts)

    with pytest.raises(embeddings.EmbeddingUnavailableError):
        documents.ingest(session_id, "sample.pdf", _make_pdf("hello"))

    assert documents.list_documents(session_id) == []
    assert documents.list_chunks(session_id) == []


def test_ingest_page_extraction_failure_raises_corrupted_no_partial_rows(monkeypatch):
    """A page-extraction exception that isn't a PdfReadError subclass (e.g.
    pypdf's ParseError/DependencyError, or any other internal failure) must
    still surface as CorruptedPDFError, not escape as a raw 500. Uses the
    injectable worker (module-level, picklable - required for the "spawn"
    isolation context) to simulate this deterministically, rather than
    monkeypatching pypdf internals, which the isolated child process
    (started fresh via spawn) would never see."""
    session_id = context.create_session()
    pdf_bytes = _make_pdf("hello", pages=1)
    monkeypatch.setattr(documents, "_extraction_worker", _worker_reports_corrupted)

    with pytest.raises(documents.CorruptedPDFError):
        documents.ingest(session_id, "sample.pdf", pdf_bytes)

    assert documents.list_documents(session_id) == []
    assert documents.list_chunks(session_id) == []


def test_ingest_embedding_unavailable_raises_no_partial_rows(monkeypatch):
    session_id = context.create_session()

    def _raise(_texts):
        raise embeddings.EmbeddingUnavailableError("model failed to load")
    monkeypatch.setattr(embeddings, "embed_texts", _raise)

    with pytest.raises(embeddings.EmbeddingUnavailableError):
        documents.ingest(session_id, "sample.pdf", _make_pdf("hello"))

    assert documents.list_documents(session_id) == []
    assert documents.list_chunks(session_id) == []


def test_ingest_unknown_session_raises_session_not_found():
    with pytest.raises(context.SessionNotFoundError):
        documents.ingest("does-not-exist", "sample.pdf", _make_pdf("hello"))


def test_delete_document_removes_row_and_chunks():
    session_id = context.create_session()
    result = documents.ingest(session_id, "sample.pdf", _make_pdf("hello"))

    documents.delete_document(result["document_id"], session_id)

    assert documents.list_documents(session_id) == []
    assert documents.list_chunks(session_id) == []


def test_delete_document_unknown_id_raises_not_found():
    session_id = context.create_session()
    with pytest.raises(documents.DocumentNotFoundError):
        documents.delete_document("does-not-exist", session_id)


def test_delete_document_wrong_session_is_not_found_not_success():
    session_a = context.create_session()
    session_b = context.create_session()
    result = documents.ingest(session_a, "sample.pdf", _make_pdf("hello"))

    with pytest.raises(documents.DocumentNotFoundError):
        documents.delete_document(result["document_id"], session_b)

    assert len(documents.list_documents(session_a)) == 1


def test_session_delete_cascades_documents_and_chunks():
    session_id = context.create_session()
    documents.ingest(session_id, "sample.pdf", _make_pdf("hello"))

    context.delete_session(session_id)

    conn = db.get_connection()
    try:
        doc_count = conn.execute("SELECT COUNT(*) FROM documents WHERE session_id = ?",
                                 (session_id,)).fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks WHERE session_id = ?",
                                   (session_id,)).fetchone()[0]
    finally:
        conn.close()
    assert doc_count == 0
    assert chunk_count == 0


def test_list_chunks_cross_session_isolation():
    session_a = context.create_session()
    session_b = context.create_session()
    documents.ingest(session_a, "a.pdf", _make_pdf("session A content"))
    documents.ingest(session_b, "b.pdf", _make_pdf("session B content"))

    chunks_a = documents.list_chunks(session_a)
    chunks_b = documents.list_chunks(session_b)

    assert all("session A" in c["content"] for c in chunks_a)
    assert all("session B" in c["content"] for c in chunks_b)
    assert len(chunks_a) == 1 and len(chunks_b) == 1


def test_list_documents_orders_by_created_at_not_random_uuid(monkeypatch):
    """documents.id is a uuid4 hex, not an autoincrement int, so ordering by
    id alone would be effectively random upload order - list_documents must
    order by created_at instead, matching the chronological-order convention
    of the other list endpoints (messages/memories)."""
    session_id = context.create_session()
    times = iter(["2026-01-01T00:00:00+00:00", "2026-01-01T00:00:01+00:00"])
    monkeypatch.setattr(documents, "_now", lambda: next(times))
    monkeypatch.setattr("uuid.uuid4", lambda: __import__("uuid").UUID(int=0xFFFFFFFF))
    documents.ingest(session_id, "first-uploaded.pdf", _make_pdf("first"))
    monkeypatch.setattr("uuid.uuid4", lambda: __import__("uuid").UUID(int=0x00000000))
    documents.ingest(session_id, "second-uploaded.pdf", _make_pdf("second"))

    docs = documents.list_documents(session_id)

    assert [d["filename"] for d in docs] == ["first-uploaded.pdf", "second-uploaded.pdf"]
