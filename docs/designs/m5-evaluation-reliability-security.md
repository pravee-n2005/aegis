# M5: Evaluation, Reliability and Security Validation

## Context

Live M4 testing (PDF upload → RAG retrieval → synthesis) surfaced one
concrete reliability bug: given a 6-page document and a question with a
clear answer spread across a few pages, the agent issued four
increasingly-paraphrased `search_documents` calls, all of which returned
genuinely relevant chunks, and then exhausted `max_tool_rounds` (4) before
ever synthesizing an answer — returning "Tool-use limit reached" instead
of a grounded response. M5's scope, per the engineering brief, was: fix
that reliability problem without weakening the round-limit safety bound,
build a deterministic/repeatable evaluation layer so retrieval and
citation quality are measured rather than assumed, and add adversarial/
security tests across the categories M4 already partially covers plus the
gaps M5 found.

## Phase 2: agent-loop reliability fix

### Root cause

Investigated in `backend/app/agent.py`, `backend/app/retrieval.py`, and
`backend/app/config.py`. Two independent gaps combined:

1. `SYSTEM_PROMPT` (`agent.py`) gave the model no stopping criterion at
   all — "use tools when helpful", nothing about recognizing sufficient
   evidence or tracking its own remaining budget.
2. `retrieval.search()`'s return shape gave the model no way to tell that
   a second, third, and fourth paraphrased query were returning
   substantially the *same* chunks as the first — each call looked
   self-contained and successful, so nothing signaled "you already have
   this."

`max_tool_rounds=4` counting every round including the synthesis round
meant that by the time the model had spent all 4 rounds searching, there
was no round left to answer in — hence the hard `tool_round_limit` cutoff
firing before synthesis.

### Fix

Two small, additive, backend-computed signals, added to every tool
result rather than a hardcoded rule:

- **`rounds_remaining`**: computed once per loop iteration in
  `agent.py::ask_aegis` (`settings.max_tool_rounds - round_num - 1`) and
  attached to every tool's result dict before it's traced/returned.
  `SYSTEM_PROMPT` now tells the model that when this hits 0 it must answer
  from what it has rather than calling another tool.
- **`new_chunk_count` / `distinct_pages_covered`**: computed in
  `agent.py::_run_tool`'s `search_documents` branch from chunk-identity
  deduplication (`trust_ctx["seen_chunk_ids"]`, a per-turn set), using the
  `chunk_id` now included in `retrieval.search()`'s return dicts
  (`retrieval.py`). A follow-up query that returns 0 new chunks gives the
  model an objective "this didn't add anything" signal instead of asking
  it to infer overlap from re-reading result text.

### Why this over the alternatives the brief explicitly ruled out

- **Not** a fixed "stop after N `search_documents` calls" rule — that's
  brittle and query-independent in the wrong direction (sometimes 1 call
  genuinely isn't enough; sometimes 1 is plenty).
- **Not** an extra LLM call to judge sufficiency — the backend already has
  deterministic chunk-identity information; spending a second model call
  to re-derive it would add latency/cost for no new information.
- **Not** a `max_tool_rounds` increase — doesn't address why the model
  doesn't know when to stop, just delays the same failure mode further
  out.
- Both signals are additive JSON fields on existing tool results and one
  added sentence in `SYSTEM_PROMPT`; no change to `TOOLS`,
  `SENSITIVE_TOOLS_AFTER_DOCUMENT_CONTENT`, skill scoping, or the chat
  API's response contract (`answer`/`trace`/`stop_reason`/`degraded_tools`
  all unchanged in shape).

### Tests

`backend/tests/test_agent.py`, "M5: agent-loop reliability" section:
sufficient evidence synthesizing within budget, insufficient evidence
correctly allowing a justified second (genuinely novel) search, a
redundant/paraphrased second search reporting `new_chunk_count == 0`, a
failing tool call still carrying `rounds_remaining`, and the existing
`tool_round_limit` regression test extended to assert `rounds_remaining`
counts down to 0 without going negative.

## Phase 3: evaluation framework

See `docs/EVALUATION.md` for the full writeup (how to run it, dataset
structure, metrics and their limitations, and the deterministic/
model-dependent boundary). Summary of the architecture:

- `backend/eval/corpus.py` — 9 hand-authored chunks, hand-crafted
  orthogonal "topic vectors" standing in for a real embedding model, so
  ranking is exact and reproducible offline.
- `backend/eval/dataset.py` — 6 questions with hand-labeled relevant
  `(filename, page)` sets.
- `backend/eval/metrics.py` — Recall@K, Precision@K, MRR, each documented
  with what they do and don't claim.
- `backend/eval/harness.py` — loads the corpus into a throwaway SQLite
  session, runs the real `retrieval.search()` (retrieval eval) and the
  real `ask_aegis()` agent loop with a scripted LLM client (grounded-
  answer/citation eval), both with `embeddings.embed_texts` faked
  deterministically.
- `backend/eval/run_eval.py` — CLI entry point (`python -m eval.run_eval`),
  human-readable or `--json` report, non-zero exit on regression.
- `backend/tests/test_eval.py` — wires the harness into `pytest -q` as a
  standing regression gate, fully offline.

A genuine live-model evaluation (real OpenRouter-backed model, real
citation phrasing) was deliberately **not** built as part of this suite,
per the constraint that ordinary `pytest` runs must never depend on a paid
API call — see `docs/EVALUATION.md`, "Deterministic vs. model-dependent
evaluation" for what that would need to look like if added later.

## Phase 4: security and adversarial testing

See `docs/SECURITY.md` for the full index of what's covered, where, and
what isn't. Most of the required categories already had solid M1-M4
coverage (malformed PDFs, oversized uploads, corrupted/invalid embeddings,
embedding-service degradation, document prompt injection against the
sensitive-tool pivot, SQL-injection-shaped session ids, cross-session
isolation at the `documents.py`/`retrieval.py`/`context.py` unit level).
M5 added:

1. **A real gap, found and fixed**: `agent.py::_run_tool` dispatched
   purely on tool name with no check against what was actually offered to
   the model that round. `scope_tools`/`scope_tools_by_names` only
   controlled the advertised `tools=` list — under a deny-all skill, a
   scripted response naming `calculator` anyway still executed it. Fixed
   with a code-level `allowed_tool_names` check in `_run_tool`, re-derived
   from the exact `tools=` list sent that round, exempting only
   `load_skill` (which must always stay callable per the existing M3
   "not a permanent lock" guarantee). Confirmed via a reproduction script
   before the fix, then pinned with three new tests.
2. **Endpoint/end-to-end gap fill**: the existing cross-session isolation
   tests all call `documents.py`/`retrieval.py` directly; added matching
   tests through the real `/api/sessions/.../documents` and `/api/chat`
   HTTP endpoints, which a unit-level regression wouldn't catch.
3. **Provider-failure gap fill**: `main.py::chat`'s `APIError → 502`
   mapping had no test.
4. **Trace content-leak gap fill**: `agent.py`'s `write_trace("tool_call",
   ...)` call was already designed to log only lengths/booleans, never
   raw tool input/output — but nothing pinned that against regression.
   Added a test that plants a secret-looking marker in document content
   and asserts it's absent from the on-disk trace file.

No existing security check was weakened or removed to make any test pass.

## Phase 5: regression and final state

Full `pytest -q` run from `backend/`: **197 passed, 0 failed** (baseline
before M5 was 181; net +16 after accounting for 3 pre-existing tests
updated for the new `rounds_remaining` field on every tool result, which
is an additive, non-breaking API change to the trace shape only).

## Known limitations and future improvements

- `max_tool_rounds`'s reliability fix is a prompt-adherence improvement on
  top of a hard backstop, not a hard guarantee — a sufficiently weak model
  (the default is a free-tier OpenRouter model) may still not act on the
  `rounds_remaining`/`new_chunk_count` signals every time; the hard
  `tool_round_limit` cutoff remains the actual safety bound and was left
  untouched.
- The evaluation suite is deterministic and offline by design, which means
  it cannot, by itself, validate real model citation behavior — see
  `docs/EVALUATION.md`'s "What this evaluation does not measure".
- `docs/SECURITY.md`'s "Known limitations" section applies in full; this
  remains a learning-starter security posture, not a production one.
- No code-coverage tooling (`pytest-cov`) was added, per the instruction
  to avoid unnecessary dependencies; critical-path coverage was instead
  verified by direct inspection against each M5 Phase 4 requirement
  category (see `docs/SECURITY.md`'s section-by-section test mapping).
