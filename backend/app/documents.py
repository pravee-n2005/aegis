"""Session-scoped document ingestion, chunking and CRUD for M4 RAG.

Sole owning module for the documents/chunks tables (db.py's convention,
mirrored from memory.py's docstring: "build_context calls
memory.list_memories() rather than querying the memories table directly,
so no second module ever learns that schema"). retrieval.py (M4 Phase 2)
calls list_chunks() here for the rows it ranks; it never queries chunks
directly.

Validation order for ingest() matters: byte-size check (cheapest, catches
a naive oversized upload) -> isolated-process PDF parse/extraction (the
actual resource boundary against a decompression bomb - see
_extract_pdf_isolated) -> per-page text-length/chunk-count caps (defense in
depth against a bomb that stays inside the extraction worker's own
resource limits but still produces an unreasonable amount of text) ->
chunking/embedding -> one transaction. A failure at any step leaves zero
rows committed - never a document row with no matching chunks.
"""
import io
import json
import multiprocessing
import sqlite3
import uuid
from datetime import datetime, timezone

from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError

from . import db
from . import embeddings
from .context import SessionNotFoundError
from .tracing import new_request_id, write_trace

try:
    import resource
except ImportError:
    resource = None  # POSIX-only (Linux/macOS); Windows falls back to the
                      # wall-clock timeout alone - process isolation still
                      # bounds the blast radius to one killable child process.

EVENT_INGESTED = "document_ingested"
EVENT_DELETED = "document_deleted"

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PAGES = 500
MAX_CHUNKS_PER_DOCUMENT = 5000
MAX_PAGE_TEXT_CHARS = 200_000
CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 150

# Resource boundary for the isolated extraction worker (security hardening,
# 2026-11-15): a wall-clock timeout the parent enforces by killing the
# child, plus (where the OS supports it) a hard memory/CPU-time cap the
# child sets on itself before touching any PDF bytes. This - not the
# MAX_PAGE_TEXT_CHARS check below, which only runs AFTER extraction has
# already materialized a page's text - is what actually bounds a
# decompression-bomb PDF's blast radius to one process the parent can kill.
PDF_EXTRACTION_TIMEOUT_SECONDS = 20
PDF_EXTRACTION_MEMORY_BYTES = 512 * 1024 * 1024


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


class PDFExtractionError(InvalidPDFError):
    """The isolated extraction worker (_extract_pdf_isolated) was killed for
    exceeding its wall-clock timeout, hit its memory/CPU resource limit, or
    exited abnormally for any other reason - the actual boundary against a
    decompression-bomb PDF. Distinct from CorruptedPDFError (a PDF that
    fails to parse cleanly within its resource budget) because this PDF may
    be well-formed, just adversarially expensive."""


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


def _extraction_worker(file_bytes: bytes, result_queue, max_pages: int,
                        memory_bytes: int, cpu_seconds: int) -> None:
    """Runs in an isolated child process (see _extract_pdf_isolated) with
    its own memory/CPU limits applied before any PDF byte is touched.
    Must never raise into the parent - every outcome, including an
    unexpected exception, is reported through the queue so the parent can
    tell a clean PDF-validation error apart from a resource-limit kill
    (which never reaches this point at all - the OS terminates the process
    directly, and the parent detects that via a nonzero/negative exitcode).

    max_pages/memory_bytes/cpu_seconds are passed in explicitly, read by
    the parent at call time, rather than referencing MAX_PAGES/
    PDF_EXTRACTION_* as bare module globals here: this process is started
    with the "spawn" method (fork-safety - see _extract_pdf_isolated), so
    it re-imports documents.py fresh and would never see a test's
    monkeypatch of those parent-process module attributes otherwise."""
    if resource is not None:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except (ValueError, OSError):
            pass  # best-effort hardening; the parent's wall-clock join() timeout is the hard backstop
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        if reader.is_encrypted:
            reader.decrypt("")
        page_count = len(reader.pages)
        if page_count > max_pages:
            result_queue.put(("page_limit", page_count))
            return
        page_texts = [page.extract_text() or "" for page in reader.pages]
        result_queue.put(("ok", page_count, page_texts))
    except FileNotDecryptedError:
        result_queue.put(("encrypted",))
    except Exception as exc:
        result_queue.put(("corrupted", str(exc)))


def _extract_pdf_isolated(file_bytes: bytes, worker=None) -> tuple[int, list[str]]:
    """Parse a PDF and extract each page's text in an isolated child
    process, returning (page_count, page_texts). This is the actual
    resource boundary against a decompression bomb: a wall-clock timeout
    the parent enforces by terminating/killing the child if it's still
    alive, plus (where the OS supports it) a memory/CPU cap the child sets
    on itself. Raises EncryptedPDFError, PageLimitExceededError,
    CorruptedPDFError, or PDFExtractionError (timeout / resource-limit
    kill / abnormal exit) - never lets a hung or memory-exhausting
    extraction run unbounded in this process.

    `worker` is injectable (default: the current value of the module-level
    _extraction_worker, looked up here rather than bound as a default
    argument, so a test's monkeypatch of documents._extraction_worker
    reaches calls made through ingest() too) so tests can simulate a hang
    or an abnormal exit deterministically, without needing a real
    malicious PDF or relying on OS resource limits firing within a test's
    time budget. It must be a module-level (picklable) callable - the
    "spawn" context below sends it to the child process by reference."""
    target = worker if worker is not None else _extraction_worker
    timeout = PDF_EXTRACTION_TIMEOUT_SECONDS
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(target=target,
                          args=(file_bytes, result_queue, MAX_PAGES,
                                PDF_EXTRACTION_MEMORY_BYTES, timeout))
    process.start()
    process.join(timeout + 5)  # grace margin beyond the child's own CPU-time rlimit
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join()
        raise PDFExtractionError(
            f"PDF extraction exceeded the {timeout}-second time limit")
    if process.exitcode != 0:
        raise PDFExtractionError(
            f"PDF extraction process terminated abnormally (exit code {process.exitcode}), "
            f"likely a resource-limit violation")
    try:
        outcome = result_queue.get_nowait()
    except Exception as exc:
        raise PDFExtractionError("PDF extraction process produced no result") from exc

    kind = outcome[0]
    if kind == "ok":
        _, page_count, page_texts = outcome
        return page_count, page_texts
    if kind == "encrypted":
        raise EncryptedPDFError("PDF appears encrypted")
    if kind == "page_limit":
        page_count = outcome[1]
        raise PageLimitExceededError(
            f"PDF has {page_count} pages, exceeds the {MAX_PAGES}-page limit")
    raise CorruptedPDFError("PDF appears corrupted")


def ingest(session_id: str, filename: str, file_bytes: bytes,
            request_id: str = None) -> dict:
    """Validate, extract, chunk, embed and store one PDF. Returns
    {"document_id", "filename", "page_count", "status"}. Raises
    UploadTooLargeError, EncryptedPDFError, CorruptedPDFError,
    PageLimitExceededError, ChunkLimitExceededError, PDFExtractionError,
    SessionNotFoundError or
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

    page_count, page_texts = _extract_pdf_isolated(file_bytes)

    chunk_rows = []  # (page_number, chunk_index, content)
    for page_number, page_text in enumerate(page_texts, start=1):
        if len(page_text) > MAX_PAGE_TEXT_CHARS:
            # Defense in depth, not the primary resource boundary: a bomb
            # that stays inside the extraction worker's own memory/CPU
            # limits (_extract_pdf_isolated) could still produce more text
            # than is reasonable to chunk/embed/store. The worker's
            # resource limits are what actually bound the blast radius of
            # a decompression bomb; this only bounds what happens next.
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
    if len(embeddings_list) != len(chunk_rows):
        # A contract violation (embed_texts() must return exactly one vector
        # per input text), not a chunk-level failure - zip() below would
        # otherwise silently truncate to the shorter list, committing a
        # document marked "ready" with fewer chunk rows than its own
        # chunk_count/page_count trace fields claim.
        raise embeddings.EmbeddingUnavailableError(
            f"embedding backend returned {len(embeddings_list)} vectors "
            f"for {len(chunk_rows)} chunks")

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
