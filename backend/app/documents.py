"""Session-scoped document ingestion, chunking and CRUD for M4 RAG.

Sole owning module for the documents/chunks tables (db.py's convention,
mirrored from memory.py's docstring: "build_context calls
memory.list_memories() rather than querying the memories table directly,
so no second module ever learns that schema"). retrieval.py (M4 Phase 2)
calls list_chunks() here for the rows it ranks; it never queries chunks
directly.

Validation order for ingest() matters: byte-size check (cheapest, catches
a naive oversized upload) -> PDF parse + page-count check (catches a
small-file "PDF bomb" before the expensive per-page extraction runs) ->
per-page extraction/chunking/embedding -> one transaction. A failure at any
step leaves zero rows committed - never a document row with no matching
chunks.
"""
import io
import json
import sqlite3
import uuid
from datetime import datetime, timezone

from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError

from . import db
from . import embeddings
from .context import SessionNotFoundError
from .tracing import new_request_id, write_trace

EVENT_INGESTED = "document_ingested"
EVENT_DELETED = "document_deleted"

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PAGES = 500
MAX_CHUNKS_PER_DOCUMENT = 5000
MAX_PAGE_TEXT_CHARS = 200_000
CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 150


class DocumentNotFoundError(Exception):
    def __init__(self, document_id):
        super().__init__(f"Unknown document_id: {document_id!r}")
        self.document_id = document_id


class UploadTooLargeError(Exception):
    """Byte-size cap exceeded - maps to 413, checked before any PDF parsing."""


class InvalidPDFError(Exception):
    """Base class for a PDF that fails validation - maps to 422."""


class EncryptedPDFError(InvalidPDFError):
    pass


class CorruptedPDFError(InvalidPDFError):
    pass


class PageLimitExceededError(InvalidPDFError):
    pass


class ChunkLimitExceededError(InvalidPDFError):
    """A dense-text PDF within the byte/page caps but producing more chunks
    than MAX_CHUNKS_PER_DOCUMENT - the byte-size and page-count caps alone
    don't bound total extracted-text volume, so a 500-page PDF with small,
    dense text per page could otherwise embed/insert an unbounded number of
    chunks in one request."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_session(conn: sqlite3.Connection, session_id: str) -> None:
    row = conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if row is None:
        raise SessionNotFoundError(session_id)


def _chunk_page_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS,
                      overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Split one page's text into overlapping char-count chunks. A chunk
    never spans a page boundary (caller passes one page's text at a time) -
    sacrifices a little chunk-boundary coherence for every chunk having one
    unambiguous page number, which citation needs more than coherence does."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += step
    return chunks


def _read_pdf(file_bytes: bytes) -> PdfReader:
    # Catches bare Exception, not just pypdf's own PdfReadError, because
    # pypdf's errors.ParseError and errors.DependencyError do NOT inherit
    # from PdfReadError (verified against pypdf 7.1.0's errors.py) - a
    # narrower catch here would let either escape as an unhandled 500 for
    # an untrusted upload that is merely malformed, not encrypted.
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            reader.decrypt("")
    except FileNotDecryptedError as exc:
        raise EncryptedPDFError("PDF appears encrypted") from exc
    except Exception as exc:
        raise CorruptedPDFError("PDF appears corrupted") from exc
    try:
        page_count = len(reader.pages)
    except FileNotDecryptedError as exc:
        raise EncryptedPDFError("PDF appears encrypted") from exc
    except Exception as exc:
        raise CorruptedPDFError("PDF appears corrupted") from exc
    if page_count > MAX_PAGES:
        raise PageLimitExceededError(
            f"PDF has {page_count} pages, exceeds the {MAX_PAGES}-page limit")
    return reader


def ingest(session_id: str, filename: str, file_bytes: bytes,
            request_id: str = None) -> dict:
    """Validate, extract, chunk, embed and store one PDF. Returns
    {"document_id", "filename", "page_count", "status"}. Raises
    UploadTooLargeError, EncryptedPDFError, CorruptedPDFError,
    PageLimitExceededError, ChunkLimitExceededError, SessionNotFoundError or
    embeddings.EmbeddingUnavailableError - never a partial commit on any of
    these."""
    rid = request_id or new_request_id()

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise UploadTooLargeError(
            f"upload is {len(file_bytes)} bytes, exceeds the {MAX_UPLOAD_BYTES}-byte limit")

    conn = db.get_connection()
    try:
        _require_session(conn, session_id)
    finally:
        conn.close()

    reader = _read_pdf(file_bytes)
    page_count = len(reader.pages)

    chunk_rows = []  # (page_number, chunk_index, content)
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            raise CorruptedPDFError("PDF appears corrupted") from exc
        if len(page_text) > MAX_PAGE_TEXT_CHARS:
            # A single pathological page (nested/repeated content streams
            # decompressing far beyond what the compressed PDF bytes imply)
            # must be rejected here, before it is ever chunked - the
            # MAX_CHUNKS_PER_DOCUMENT check below only catches a bomb spread
            # across many chunks, not one page whose own extracted text is
            # already enormous.
            raise ChunkLimitExceededError(
                f"page {page_number} produces {len(page_text)} characters of text, "
                f"exceeds the {MAX_PAGE_TEXT_CHARS}-character per-page limit")
        for chunk_index, chunk_text in enumerate(_chunk_page_text(page_text)):
            chunk_rows.append((page_number, chunk_index, chunk_text))
            if len(chunk_rows) > MAX_CHUNKS_PER_DOCUMENT:
                raise ChunkLimitExceededError(
                    f"PDF produces more than {MAX_CHUNKS_PER_DOCUMENT} chunks, "
                    f"exceeds the per-document chunk limit")

    embeddings_list = embeddings.embed_texts([content for _, _, content in chunk_rows])

    document_id = uuid.uuid4().hex
    status = "ready" if chunk_rows else "empty"
    now = _now()

    conn = db.get_connection()
    try:
        _require_session(conn, session_id)
        conn.execute(
            "INSERT INTO documents (id, session_id, filename, page_count, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (document_id, session_id, filename, page_count, status, now),
        )
        if chunk_rows:
            conn.executemany(
                "INSERT INTO chunks (document_id, session_id, page_number, chunk_index, "
                "content, embedding, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (document_id, session_id, page_number, chunk_index, content,
                     json.dumps(vector), now)
                    for (page_number, chunk_index, content), vector
                    in zip(chunk_rows, embeddings_list)
                ],
            )
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        raise
    finally:
        conn.close()

    write_trace(EVENT_INGESTED, rid, session_id=session_id, document_id=document_id,
                filename_length=len(filename), page_count=page_count,
                chunk_count=len(chunk_rows), status=status)
    return {"document_id": document_id, "filename": filename,
            "page_count": page_count, "status": status}


def list_documents(session_id: str) -> list:
    conn = db.get_connection()
    try:
        _require_session(conn, session_id)
        rows = conn.execute(
            "SELECT id, filename, page_count, status, created_at FROM documents "
            "WHERE session_id = ? ORDER BY created_at ASC, id ASC",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()
    return [
        {"document_id": r["id"], "filename": r["filename"], "page_count": r["page_count"],
         "status": r["status"], "created_at": r["created_at"]}
        for r in rows
    ]


def delete_document(document_id: str, session_id: str, request_id: str = None) -> None:
    conn = db.get_connection()
    try:
        cursor = conn.execute(
            "DELETE FROM documents WHERE id = ? AND session_id = ?",
            (document_id, session_id),
        )
        conn.commit()
    finally:
        conn.close()
    if cursor.rowcount == 0:
        raise DocumentNotFoundError(document_id)
    write_trace(EVENT_DELETED, request_id or new_request_id(),
                session_id=session_id, document_id=document_id)


def list_chunks(session_id: str) -> list:
    """Every chunk row for a session, across all its documents - the sole
    read path retrieval.py uses; it never queries the chunks table itself."""
    conn = db.get_connection()
    try:
        rows = conn.execute(
            "SELECT c.id, c.document_id, d.filename, c.page_number, c.content, c.embedding "
            "FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE c.session_id = ? ORDER BY c.id ASC",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()
    return [
        {"id": r["id"], "document_id": r["document_id"], "filename": r["filename"],
         "page_number": r["page_number"], "content": r["content"], "embedding": r["embedding"]}
        for r in rows
    ]
