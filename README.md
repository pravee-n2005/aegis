# AEGIS — Engineering Intelligence Workbench

Starter backend for Milestone 1: a Claude-powered assistant with a bounded tool-use loop, safe calculator, read-only local workspace search, and execution trace. Milestone 4 adds document upload and retrieval-augmented question answering, grounded with filename+page citations. Milestone 5 adds a deterministic offline evaluation suite for retrieval/citation quality, an agent-loop reliability fix for the repeated-search problem observed in live M4 testing, and additional adversarial/security tests — see [docs/EVALUATION.md](docs/EVALUATION.md) and [docs/SECURITY.md](docs/SECURITY.md).

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
Add your [OpenRouter](https://openrouter.ai/keys) API key to `.env` as `OPENROUTER_API_KEY`. The app talks to OpenRouter through the Anthropic Python SDK's Anthropic-Messages-compatible endpoint (`base_url=https://openrouter.ai/api`), not Anthropic's own API directly. `OPENROUTER_MODEL` defaults to a free OpenRouter model (`nvidia/nemotron-3-ultra-550b-a55b:free` at the time this was written — OpenRouter's free-model lineup rotates, so check [openrouter.ai/models?q=free](https://openrouter.ai/models?q=free) if it stops resolving) and can be set to any OpenRouter model ID enabled for your account.

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
This includes `tests/test_eval.py`, a deterministic regression gate wired from the M5 evaluation suite — fully offline (fake embeddings, scripted LLM), no API key or model download needed. Run the standalone evaluation report (same harness, human-readable or `--json` output) with:
```bash
python -m eval.run_eval
```
See [docs/EVALUATION.md](docs/EVALUATION.md) for what it measures, its limitations, and how to add cases.

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

### Document search behavior

`search_documents` only returns chunks scoring at or above a cosine-similarity
relevance threshold (default `0.2`) — an unrelated query returns no matches at
all rather than the nearest-but-irrelevant chunks, so the model has nothing
weak to cite from. Override the default with the `SEARCH_RELEVANCE_THRESHOLD`
env var if your documents need a different cutoff.

`search_documents` participates in the same skill tool-scoping as
`calculator` and `search_workspace` — it is **not** unconditionally
available. If a loaded skill's `allowed-tools` list omits it (or sets
`allowed-tools: []`), document search is blocked for that skill, same as any
other tool.

PDF text extraction runs in an isolated child process, not inline in the
request, so a pathological PDF can't tie up the server indefinitely:
extraction is killed if it exceeds a wall-clock timeout, and — on operating
systems that support it (Linux/macOS) — the child process also sets its own
hard memory and CPU-time limits before it touches any PDF bytes. A timeout,
a resource-limit kill, or any other extraction failure returns `422`, the
same status as other malformed-PDF rejections. There's also a per-page
extracted-text character cap, but that's defense in depth for text that's
already been safely extracted — it is not what bounds a decompression
bomb's resource usage; the process isolation above is.

If the local embedding model is unavailable when `search_documents` runs
mid-chat, the tool returns a generic "temporarily unavailable" error (never
an internal file path or stack trace) and the chat response's
`degraded_tools` field lists which tool call failed that way, e.g.
`"degraded_tools": ["search_documents"]` (normally `[]`). Check this field if
you need to know programmatically whether an answer might be based on
incomplete evidence, rather than relying on the model's wording alone.

None of this amounts to complete prompt-injection protection or
production-grade multi-tenant security — see the caveats above and in
`docs/designs/m4-rag-document-intelligence.md` and `docs/SECURITY.md`.

### Agent-loop reliability (M5)

Live M4 testing showed the agent spending all 4 tool rounds on paraphrased
`search_documents` queries against the same small document, then hitting the
round limit before ever synthesizing an answer. Every tool result now carries
a backend-computed `rounds_remaining` count, and `search_documents` results
additionally carry `new_chunk_count` / `distinct_pages_covered` — objective
signals (never a hardcoded "one search is enough" rule) that let the model
recognize when a follow-up query is actually redundant versus genuinely
surfacing new evidence. `max_tool_rounds` (default `4`) is unchanged and
remains the hard safety bound. See `docs/designs/m5-evaluation-reliability-security.md`.

Never commit `.env`. Workspace search is read-only and limited to Markdown/text files under `backend/workspace`. Do not expose the API publicly without authentication, rate limits, and authorization.
