from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from anthropic import APIError
from .agent import ask_aegis, TOOLS
from .config import settings
from .skills import discover_skills, EVENT_DISCOVERED
from .tracing import new_request_id, redact, write_trace

app = FastAPI(title="AEGIS API", version="0.1.0")

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    history: list[dict] = Field(default_factory=list)

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "aegis-api"}

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
    try:
        return ask_aegis(request.message, request.history)
    except APIError as exc:
        raise HTTPException(status_code=502,
                            detail=f"LLM provider error: {exc.__class__.__name__}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
