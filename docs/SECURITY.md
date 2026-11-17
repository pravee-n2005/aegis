# AEGIS Security Test Coverage (through M5)

This is an index of the adversarial/security guarantees AEGIS currently
tests and enforces, where they live, and what is explicitly **not**
covered. It supplements the inline security-hardening comments in
`backend/app/agent.py`, `backend/app/documents.py`, and
`backend/docs/designs/m4-rag-document-intelligence.md` rather than
replacing them — read the code comments for the full reasoning behind
each decision; this file is the map.

AEGIS is a learning starter, not a production-hardened multi-tenant
service. None of the below amounts to defense against a sophisticated,
motivated attacker with direct database or process access — see "Known
limitations" at the end.

## Document prompt injection

**Guarantee**: content retrieved from an uploaded, attacker-controllable
PDF cannot cause the backend to execute `search_workspace` (which reads
local workspace files) for the rest of that turn, even if the model is
"successfully injected" into requesting it.

**Mechanism**: `SENSITIVE_TOOLS_AFTER_DOCUMENT_CONTENT` gate in
`app/agent.py::_run_tool`, keyed on `trust_ctx["document_content_seen"]`.
Code-level, not prompt-level — enforced regardless of what the model's
own text says.

**Tests**: `tests/test_agent.py::test_malicious_document_content_cannot_invoke_search_workspace`,
`::test_search_workspace_unaffected_when_no_document_content_retrieved` (the gate doesn't
over-trigger on legitimate use).

**Explicit limitation**: this only closes the "pivot to a sensitive tool"
path. It does **not** prevent the model's own prose answer from being
influenced by injected instructions, and any future sensitive tool not
added to `SENSITIVE_TOOLS_AFTER_DOCUMENT_CONTENT` is not covered by it.

## Tool use outside the permitted skill scope (M5)

**Guarantee**: a tool call naming a tool outside the scope the model was
actually offered this round is never executed, regardless of why the
model named it (compromised/non-compliant provider response, model error,
or a document successfully steering the model to request a tool it was
never offered).

**Mechanism** (added in M5 — this closed a real gap found while writing
this suite): `app/agent.py::_run_tool`'s `allowed_tool_names` check,
computed once per round from the exact `tools=` list sent to that round's
API call and re-checked in code against every `tool_use` block in the
response. Previously, `scope_tools`/`scope_tools_by_names` only controlled
what was *advertised* to the model (the `tools=` parameter) — the
dispatcher itself had no allowlist check, so a tool_use for a tool that
was never offered still executed if one somehow arrived.

**Tests**: `tests/test_agent.py::test_tool_call_outside_deny_all_skill_scope_is_not_executed`,
`::test_tool_call_excluded_by_partial_allowlist_is_not_executed`,
`::test_load_skill_still_works_under_its_own_deny_all_scope` (regression: the
gate must not block the one tool that must always stay callable).

## Session isolation (documents and memories)

**Guarantee**: one session's documents, chunks, and memories are never
visible to, retrievable by, or deletable from a different session, even
though all sessions share one SQLite database file and one running
process.

**Mechanism**: every query and write in `documents.py`, `retrieval.py`,
`memory.py`, and `context.py` filters by `session_id`; foreign keys with
`ON DELETE CASCADE` ensure a deleted session's rows are fully removed
across `messages`/`summaries`/`memories`/`documents`/`chunks`.

**Tests** (unit level): `tests/test_documents.py::test_list_chunks_cross_session_isolation`,
`::test_session_delete_cascades_documents_and_chunks`,
`::test_delete_document_wrong_session_is_not_found_not_success`;
`tests/test_context.py::test_session_isolation_build_context_and_memories_never_cross`;
`tests/test_retrieval.py::test_search_cross_session_isolation`.

**Tests** (API/end-to-end level, added in M5 — the unit tests above call
`documents.py`/`retrieval.py` directly and would not catch a routing bug
at the endpoint layer): `tests/test_documents_endpoints.py::test_delete_document_from_different_session_returns_404_not_success`,
`::test_chat_in_one_session_cannot_retrieve_another_sessions_document_content` (drives the
real `/api/chat` endpoint and the real agent loop, asserts `search_documents`
returns `[]` for a session that never uploaded the queried document).

## Malformed PDFs and extraction failures

**Guarantee**: an encrypted, corrupted, oversized, page-limit-exceeding,
chunk-limit-exceeding, or adversarially slow/memory-hungry ("decompression
bomb") PDF is rejected with a clean `422`/`413`, and **never** leaves a
partial document/chunk row committed.

**Mechanism**: `documents.py::ingest`'s validation order (byte size → isolated-process
extraction with wall-clock timeout + `RLIMIT_AS`/`RLIMIT_CPU` → per-page
text-length cap → chunk-count cap → one transaction). PDF extraction runs
in an isolated child process (`multiprocessing`, `spawn` context) so a
hung or resource-exhausting extraction can be killed without taking down
the request-handling process.

**Tests**: `tests/test_documents.py::test_ingest_malformed_pdf_raises_corrupted_no_partial_rows`,
`::test_ingest_page_limit_exceeded_raises_no_partial_rows`,
`::test_ingest_chunk_limit_exceeded_raises_no_partial_rows`,
`::test_ingest_single_page_text_bomb_raises_before_chunking`,
`::test_ingest_extraction_timeout_kills_worker_raises_no_partial_rows`,
`::test_ingest_extraction_abnormal_exit_raises_no_partial_rows`, plus the
matching `422` endpoint tests in `tests/test_documents_endpoints.py`.

## Oversized uploads and streaming limits

**Guarantee**: an upload whose body exceeds `MAX_UPLOAD_BYTES` (20MB) is
rejected with `413` **before** the full body is buffered in memory or
handed to PDF parsing.

**Mechanism**: `main.py::_read_upload_bounded` reads the upload in
`UPLOAD_READ_CHUNK_BYTES` (1MB) increments and aborts as soon as the
running total exceeds the limit — the streaming abort itself is the
protection; a post-hoc `len(file_bytes) > MAX_UPLOAD_BYTES` check after
buffering the whole body would be the DoS vector this is meant to avoid.

**Tests**: `tests/test_documents_endpoints.py::test_upload_document_oversized_returns_413_before_extraction`;
`tests/test_documents.py::test_ingest_oversized_upload_raises_before_parsing`
(the `ingest()`-level byte check, defense in depth behind the streaming check).

## Invalid or corrupted embedding data

**Guarantee**: a stored chunk embedding that is corrupted JSON, wrong
shape, wrong type, or contains non-finite values (`NaN`/`Infinity`) is
skipped during retrieval — logged via a trace event, never crashes the
search, never silently corrupts the ranking with a `NaN` score or a
`zip()`-truncated wrong-but-plausible comparison.

**Mechanism**: `retrieval.py::_cosine`'s explicit shape/type/finiteness
check, raising `ValueError` that `retrieval.py::search` catches per-row
and reports as `CORRUPTED_EMBEDDING_SKIPPED`, continuing with the
remaining rows. On ingest, `documents.py::ingest` raises
`embeddings.EmbeddingUnavailableError` rather than committing if the
embedder returns the wrong vector count for the input batch, instead of
letting `zip()` silently truncate the committed rows.

**Tests**: `tests/test_retrieval.py::test_search_skips_corrupted_embedding_row_without_crashing`,
`::test_search_skips_shape_mismatched_embedding_without_wrong_score`,
`::test_search_skips_non_finite_embedding_values`;
`tests/test_documents.py::test_ingest_embedder_vector_count_mismatch_raises_no_partial_rows`.

## Embedding service failures and degraded-tool reporting

**Guarantee**: if the local embedding backend is unavailable (missing
dependency, corrupted HuggingFace cache, model load failure, inference
failure) when `search_documents` runs mid-chat, the tool returns a fixed,
sanitized error message — **never** `str(exc)`, which can carry a
HuggingFace cache path or import-traceback fragment — and the chat
response's `degraded_tools` field names the affected tool so a caller can
tell programmatically that an answer may be based on incomplete evidence,
without having to parse the model's prose.

**Mechanism**: `embeddings.py::EmbeddingUnavailableError`; the fixed
message + `"degraded": True` branch in `agent.py::_run_tool`; the same
sanitization pattern in `main.py::upload_document`'s `EmbeddingUnavailableError`
handler (`503`, fixed detail string).

**Tests**: `tests/test_agent.py::test_embedding_failure_sanitized_and_flagged_degraded`
(asserts the leaky path fragment is NOT in the returned error, and that
`degraded_tools == ["search_documents"]`); `tests/test_documents_endpoints.py::test_upload_document_embedding_unavailable_returns_503_no_partial_rows`.

## Provider failures and bounded agent execution

**Guarantee**: an upstream LLM-provider failure (OpenRouter/Anthropic SDK
`APIError`) during the main chat call maps to a clean `502`, never a bare
`500` that could leak an unhandled traceback; and regardless of model
behavior, the agent loop never runs more than `max_tool_rounds` (default
4) rounds of tool use — hitting that limit returns a clean, user-facing
"Tool-use limit reached" message rather than running unbounded.

**Mechanism**: `main.py::chat`'s `except APIError` handler (added test
coverage in M5); the fixed `for round_num in range(settings.max_tool_rounds)`
loop bound in `agent.py::ask_aegis`, unchanged by the M5 reliability fix —
`rounds_remaining`/`new_chunk_count` are *signals* surfaced to the model
to encourage earlier synthesis, not a change to the hard bound itself.

**Tests**: `tests/test_main.py::test_chat_provider_api_error_maps_to_502` (M5);
`tests/test_agent.py::test_max_tool_round_exhaustion`,
`::test_max_round_exhaustion_shows_decreasing_rounds_remaining` (M5, pins
that the new signal doesn't weaken the bound).

## Sensitive data exposure through errors and traces

**Guarantee**: the on-disk execution trace (`app/tracing.py::write_trace`,
appended to `.traces/traces.jsonl`) never contains raw tool input/output
content — only lengths, booleans, and small structured metadata. Error
messages returned to API callers never leak internal file paths or
tracebacks (see embedding-service section above).

**Mechanism**: `agent.py`'s `write_trace("tool_call", ...)` call logs
`input_length`/`result_length` (via `tracing.content_length`) instead of
the input/result dicts themselves; `tracing.py::redact` bounds any field
that must be logged as text to `MAX_FIELD_CHARS`.

**Tests**: `tests/test_tracing.py::test_redact_truncates_long_values` and
related; `tests/test_agent.py::test_tool_call_trace_record_never_contains_raw_document_content`
(M5 — inserts a secret-marker string into a chunk's content, confirms it's
present in the in-process result but absent from the on-disk trace file,
and that every `tool_call` trace record has exactly the expected
length/boolean fields, no `input`/`result` keys).

**Explicit limitation**: `tracing.redact` only *truncates* length — it has
no secret-detection logic. A short secret embedded in a field that IS
intentionally logged as text (there are currently none that log arbitrary
user/document content) would not be caught by truncation alone.

## Known limitations (read before relying on any of the above)

- This is a learning-starter security posture, not a production one. No
  authentication, authorization, rate limiting, or network-level isolation
  exists at any layer tested here — see the README's final paragraph.
- The skill-scope gate (M5) and the document-content gate both operate
  within a single chat turn's in-process state (`trust_ctx`,
  `allowed_tool_names`); neither is a persistent, cryptographically
  enforced capability system.
- Prompt injection protection is scoped to the one documented pivot path
  (sensitive tools after document content); it is not a general defense
  against a model's answer text being influenced by injected content.
- All of the above was validated with the current, free-tier OpenRouter
  model configured in this starter and scripted/mocked LLM responses in
  tests. A different, more "cooperative to injection" model, or a
  genuinely adversarial response from a misbehaving/compromised provider,
  is the scenario the M5 skill-scope-enforcement fix specifically targets
  — but it has not been validated against a real adversarial model in
  live traffic, only against scripted test responses that simulate one.
