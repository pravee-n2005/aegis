# AEGIS — Engineering Intelligence Workbench

Starter backend for Milestone 1: a Claude-powered assistant with a bounded tool-use loop, safe calculator, read-only local workspace search, and execution trace. Milestone 4 adds document upload and retrieval-augmented question answering, grounded with filename+page citations.

This is a learning starter, not a production-ready SaaS. Authentication, multi-tenancy, frontend, LangGraph, and MCP are later milestones.

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

The first PDF upload downloads the local embedding model (`sentence-transformers/all-MiniLM-L6-v2`, ~90MB) to `~/.cache/huggingface`. This needs network access once; every upload after that runs fully offline.

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

Upload a PDF and ask a question grounded in it — the `session_id` from the upload response ties the question to the same session's documents:
```bash
SESSION_ID=$(curl -s -X POST http://127.0.0.1:8000/api/sessions | python3 -c 'import json,sys;print(json.load(sys.stdin)["session_id"])')

curl -X POST "http://127.0.0.1:8000/api/sessions/$SESSION_ID/documents" \
  -F "file=@sample.pdf"

curl -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SESSION_ID\",\"message\":\"What does the document say about X?\"}"
# "answer" cites the matching (filename, page) pair the search_documents tool
# actually returned, e.g. "...according to sample.pdf, page 2."
```

Never commit `.env`. Workspace search is read-only and limited to Markdown/text files under `backend/workspace`. Do not expose the API publicly without authentication, rate limits, and authorization.
