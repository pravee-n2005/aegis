from contextlib import asynccontextmanager
from sqlite3 import OperationalError
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from anthropic import APIError

from . import context
from . import memory
from .agent import ask_aegis, TOOLS
from .config import settings
from .context import SessionNotFoundError
from .memory import MemoryNotFoundError
from .skills import discover_skills, EVENT_DISCOVERED
from .tracing import new_request_id, redact, write_trace
from . import db

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
