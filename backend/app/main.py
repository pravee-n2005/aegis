import asyncio
from contextlib import asynccontextmanager
from sqlite3 import OperationalError
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field
from anthropic import APIError

from . import context
from . import documents
from . import memory
from .agent import ask_aegis, TOOLS
from .config import settings
from .context import SessionNotFoundError
from .documents import DocumentNotFoundError, InvalidPDFError, UploadTooLargeError
from .embeddings import EmbeddingUnavailableError
from .memory import MemoryNotFoundError
from .skills import discover_skills, EVENT_DISCOVERED
from .tracing import new_request_id, redact, write_trace
from . import db

UPLOAD_READ_CHUNK_BYTES = 1024 * 1024

@asynccontextmanager
async def _lifespan(app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="AEGIS API", version="0.1.0", lifespan=_lifespan)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    session_id: Optional[str] = None
    history: list[dict] = Field(default_factory=list)


@app.get("/api/health")
def health():
    try:
        conn = db.get_connection()
        conn.execute("SELECT 1")
        conn.close()
        return {"status": "ok", "service": "aegis-api", "db": "ok"}
    except (OperationalError, OSError) as exc:
        return {"status": "degraded", "service": "aegis-api", "db": "unreachable",
                "error": redact(str(exc))}


@app.get("/api/skills")
def list_skills():
    """Read-only skill metadata (name, description, allowed-tools). Never
    exposes instruction bodies. Degrades to an empty list on a discovery
    error, matching the chat path's graceful fall-through to M1 behavior —
    never a 500 for what is functionally "no skills available"."""
    request_id = new_request_id()
    known_tools = {tool["name"] for tool in TOOLS}
    try:
        skills = discover_skills(settings.skills_dir, known_tools, request_id)
    except Exception as exc:
        write_trace(EVENT_DISCOVERED, request_id, reason="PARSE_ERROR",
                    error=redact(str(exc)))
        skills = []
    return {"skills": [
        {"name": s.name, "description": s.description, "allowed_tools": s.allowed_tools}
        for s in skills
    ]}


@app.post("/api/chat")
def chat(request: ChatRequest):
    request_id = new_request_id()
    if request.session_id == "":
        raise HTTPException(status_code=400, detail="session_id must be non-empty or omitted")

    session_id = request.session_id
    seed_history = None
    try:
        if session_id is None:
            session_id = context.create_session(request_id=request_id)
            seed_history = request.history or None
        elif request.history:
            write_trace("history_ignored", request_id, reason="SESSION_ID_WINS", session_id=session_id)

        result = ask_aegis(session_id, request.message, seed_history)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="unknown session_id") from exc
    except APIError as exc:
        raise HTTPException(status_code=502,
                            detail=f"LLM provider error: {exc.__class__.__name__}")
    except (OperationalError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Storage temporarily unavailable") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    result["session_id"] = session_id
    return result


@app.post("/api/sessions")
def create_session():
    try:
        session_id = context.create_session()
    except (OperationalError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Storage temporarily unavailable") from exc
    return {"session_id": session_id}


@app.get("/api/sessions/{session_id}/messages")
def get_session_messages(session_id: str):
    try:
        messages = context.get_messages(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="unknown session_id") from exc
    except (OperationalError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Storage temporarily unavailable") from exc
    return {"messages": messages}


@app.get("/api/sessions/{session_id}/memories")
def get_session_memories(session_id: str):
    try:
        if not context.session_exists(session_id):
            raise HTTPException(status_code=404, detail="unknown session_id")
        memories = memory.list_memories(session_id)
    except (OperationalError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Storage temporarily unavailable") from exc
    return {"memories": memories}


@app.delete("/api/sessions/{session_id}/memories/{memory_id}")
def delete_session_memory(session_id: str, memory_id: int):
    try:
        memory.delete_memory(session_id, memory_id)
    except MemoryNotFoundError as exc:
        raise HTTPException(status_code=404, detail="memory not found") from exc
    except (OperationalError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Storage temporarily unavailable") from exc
    return {"status": "deleted"}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    try:
        context.delete_session(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="unknown session_id") from exc
    except (OperationalError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Storage temporarily unavailable") from exc
    return {"status": "deleted"}


async def _read_upload_bounded(file: UploadFile, max_bytes: int) -> bytes:
    """Stream-read an upload in fixed-size chunks and abort as soon as the
    total exceeds max_bytes, instead of reading the whole body into memory
    first and checking its length after - the latter is itself the DoS
    vector an oversized upload is supposed to be rejected before causing."""
    chunks = []
    total = 0
    while True:
        chunk = await file.read(UPLOAD_READ_CHUNK_BYTES)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            raise UploadTooLargeError(
                f"upload is over {max_bytes} bytes, exceeds the {max_bytes}-byte limit")
    return b"".join(chunks)


@app.post("/api/sessions/{session_id}/documents")
async def upload_document(session_id: str, file: UploadFile):
    request_id = new_request_id()
    try:
        file_bytes = await _read_upload_bounded(file, documents.MAX_UPLOAD_BYTES)
        result = await asyncio.to_thread(
            documents.ingest, session_id, file.filename, file_bytes, request_id=request_id)
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except InvalidPDFError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except EmbeddingUnavailableError as exc:
        # A fixed, greppable detail string here, not str(exc) - the
        # underlying message can carry internal detail (a HuggingFace cache
        # path, an import traceback fragment) that shouldn't reach a client.
        raise HTTPException(status_code=503,
                             detail="embedding service unavailable, try again shortly") from exc
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="unknown session_id") from exc
    except (OperationalError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Storage temporarily unavailable") from exc
    return result


@app.get("/api/sessions/{session_id}/documents")
def list_session_documents(session_id: str):
    try:
        docs = documents.list_documents(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="unknown session_id") from exc
    except (OperationalError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Storage temporarily unavailable") from exc
    return {"documents": docs}


@app.delete("/api/sessions/{session_id}/documents/{document_id}")
def delete_session_document(session_id: str, document_id: str):
    try:
        documents.delete_document(document_id, session_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail="document not found") from exc
    except (OperationalError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Storage temporarily unavailable") from exc
    return {"status": "deleted"}
