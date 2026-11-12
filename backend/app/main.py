from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from anthropic import APIError
from .agent import ask_aegis

app = FastAPI(title="AEGIS API", version="0.1.0")

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    history: list[dict] = Field(default_factory=list)

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "aegis-api"}

@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        return ask_aegis(request.message, request.history)
    except APIError as exc:
        raise HTTPException(status_code=502,
                            detail=f"LLM provider error: {exc.__class__.__name__}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
