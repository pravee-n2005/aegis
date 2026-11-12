# AEGIS — Engineering Intelligence Workbench

Starter backend for Milestone 1: a Claude-powered assistant with a bounded tool-use loop, safe calculator, read-only local workspace search, and execution trace.

This is a learning starter, not a production-ready SaaS. RAG, vector storage, authentication, multi-tenancy, frontend, LangGraph, MCP, and PyTorch experiments are later milestones.

## Fedora setup
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```
Add your Anthropic API key to `.env`. Claude Pro does not include API credits; API usage is billed separately. Set `ANTHROPIC_MODEL` to a model ID enabled for your API account.

Run:
```bash
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs

Test:
```bash
pytest -q
```

Example request:
```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Find the AEGIS project goals in the workspace and summarise them."}'
```

Never commit `.env`. Workspace search is read-only and limited to Markdown/text files under `backend/workspace`. Do not expose the API publicly without authentication, rate limits, and authorization.
