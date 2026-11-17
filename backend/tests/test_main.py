import os
import types
from sqlite3 import OperationalError
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

from app import context, db, tracing  # noqa: E402
from app.config import settings  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


def _text_block(text):
    return types.SimpleNamespace(type="text", text=text, model_dump=lambda: {"type": "text", "text": text})


def _response(content, stop_reason="end_turn"):
    return types.SimpleNamespace(content=content, stop_reason=stop_reason)


def _mock_client(responses):
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = responses
    return mock_client


@pytest.fixture(autouse=True)
def isolate_trace_and_skills(tmp_path, monkeypatch):
    monkeypatch.setattr(tracing, "TRACE_PATH", tmp_path / "traces.jsonl")
    monkeypatch.setattr(settings, "skills_dir", str(tmp_path / "skills"))
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "aegis.db"))
    db.init_db()
    return tmp_path / "skills"


def _write_skill(skills_dir, dirname, name, description, allowed_tools=None):
    skill_dir = skills_dir / dirname
    skill_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = f"name: {name}\ndescription: {description}"
    if allowed_tools is not None:
        tools_yaml = "\n".join(f"  - {t}" for t in allowed_tools)
        frontmatter += f"\nallowed-tools:\n{tools_yaml}"
    (skill_dir / "SKILL.md").write_text(f"---\n{frontmatter}\n---\nSecret body text.", encoding="utf-8")


def test_list_skills_empty_when_no_skills_directory():
    response = client.get("/api/skills")
    assert response.status_code == 200
    assert response.json() == {"skills": []}


def test_list_skills_happy_path(isolate_trace_and_skills):
    _write_skill(isolate_trace_and_skills, "demo", "demo-skill", "A demo skill.",
                 allowed_tools=["calculator"])
    response = client.get("/api/skills")
    assert response.status_code == 200
    body = response.json()
    assert body == {"skills": [
        {"name": "demo-skill", "description": "A demo skill.", "allowed_tools": ["calculator"]}
    ]}
    assert "Secret body text." not in response.text


def test_list_skills_degrades_to_empty_list_on_discovery_error(isolate_trace_and_skills, monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("simulated discovery failure")

    import app.main as main_module
    monkeypatch.setattr(main_module, "discover_skills", _raise)
    response = client.get("/api/skills")
    assert response.status_code == 200
    assert response.json() == {"skills": []}


# --- /api/health ---

def test_health_ok():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["db"] == "ok"


def test_health_degrades_when_db_unreachable(monkeypatch):
    def _raise():
        raise OperationalError("database is locked")

    monkeypatch.setattr(db, "get_connection", _raise)
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["db"] == "unreachable"


# --- POST /api/sessions, GET .../messages, .../memories ---

def test_create_session_endpoint_returns_new_id():
    response = client.post("/api/sessions")
    assert response.status_code == 200
    session_id = response.json()["session_id"]
    assert isinstance(session_id, str) and session_id


def test_create_session_endpoint_503_when_db_unavailable(monkeypatch):
    def _raise():
        raise OperationalError("database is locked")

    monkeypatch.setattr(db, "get_connection", _raise)
    response = client.post("/api/sessions")
    assert response.status_code == 503


def test_get_messages_returns_ordered_messages():
    session_id = context.create_session()
    context.append_message(session_id, "user", "first")
    context.append_message(session_id, "assistant", "second")
    response = client.get(f"/api/sessions/{session_id}/messages")
    assert response.status_code == 200
    roles = [m["role"] for m in response.json()["messages"]]
    assert roles == ["user", "assistant"]


def test_get_messages_unknown_session_404():
    response = client.get("/api/sessions/does-not-exist/messages")
    assert response.status_code == 404


def test_get_messages_zero_messages_returns_empty_list():
    session_id = context.create_session()
    response = client.get(f"/api/sessions/{session_id}/messages")
    assert response.status_code == 200
    assert response.json() == {"messages": []}


def test_get_memories_unknown_session_404():
    response = client.get("/api/sessions/does-not-exist/memories")
    assert response.status_code == 404


def test_delete_memory_endpoint_removes_and_404s_on_repeat():
    session_id = context.create_session()
    from app import memory
    memory.extract_memories(session_id, "remember that I like tea")
    [m] = memory.list_memories(session_id)

    response = client.delete(f"/api/sessions/{session_id}/memories/{m['id']}")
    assert response.status_code == 200

    response = client.delete(f"/api/sessions/{session_id}/memories/{m['id']}")
    assert response.status_code == 404


# --- DELETE /api/sessions/{id}: cascade + double-delete ---

def test_delete_session_cascades_all_tables(monkeypatch):
    """Closes Outside Voice finding 5 (T10): asserts the child rows are
    actually gone, not just that the endpoint returns 200."""
    session_id = context.create_session()
    context.append_message(session_id, "user", "hi")
    from app import memory
    memory.extract_memories(session_id, "remember that I like tea")
    conn = db.get_connection()
    conn.execute(
        "INSERT INTO summaries (session_id, content, created_at) VALUES (?, ?, ?)",
        (session_id, '{"text": "s"}', "now"),
    )
    conn.commit()
    conn.close()

    response = client.delete(f"/api/sessions/{session_id}")
    assert response.status_code == 200

    conn = db.get_connection()
    for table in ("messages", "summaries", "memories"):
        row = conn.execute(f"SELECT COUNT(*) AS c FROM {table} WHERE session_id = ?", (session_id,)).fetchone()
        assert row["c"] == 0, f"{table} still has rows after session delete"
    session_row = conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
    conn.close()
    assert session_row is None


def test_delete_session_double_delete_404s_on_second_call():
    session_id = context.create_session()
    assert client.delete(f"/api/sessions/{session_id}").status_code == 200
    assert client.delete(f"/api/sessions/{session_id}").status_code == 404


def test_delete_session_unavailable_db_maps_to_503(monkeypatch):
    session_id = context.create_session()

    def _raise(*args, **kwargs):
        raise OperationalError("database is locked")

    monkeypatch.setattr(context, "delete_session", _raise)
    response = client.delete(f"/api/sessions/{session_id}")
    assert response.status_code == 503


# --- session isolation across every endpoint (security-critical) ---

def test_session_isolation_across_endpoints():
    session_a = context.create_session()
    session_b = context.create_session()
    context.append_message(session_a, "user", "A's secret")

    # B cannot see A's messages via its own id
    response = client.get(f"/api/sessions/{session_b}/messages")
    assert response.json() == {"messages": []}

    # deleting B never touches A
    client.delete(f"/api/sessions/{session_b}")
    response = client.get(f"/api/sessions/{session_a}/messages")
    assert len(response.json()["messages"]) == 1


def test_sql_injection_shaped_session_id_404s_not_executed():
    crafted = "abc'; DROP TABLE sessions; --"
    response = client.get(f"/api/sessions/{crafted}/messages")
    assert response.status_code == 404
    # real sessions table must still be usable afterward
    assert client.post("/api/sessions").status_code == 200


# --- POST /api/chat: session continuity, history handling, OSError mapping ---

@patch("app.agent.Anthropic")
def test_chat_without_session_id_creates_one(mock_anthropic):
    mock_anthropic.return_value = _mock_client([_response([_text_block("Hi there.")])])
    response = client.post("/api/chat", json={"message": "Hello"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Hi there."
    assert isinstance(body["session_id"], str) and body["session_id"]


@patch("app.agent.Anthropic")
def test_chat_second_call_same_session_sees_first_turn_context(mock_anthropic):
    mock_anthropic.return_value = _mock_client([
        _response([_text_block("First answer.")]),
        _response([_text_block("Second answer.")]),
    ])
    first = client.post("/api/chat", json={"message": "Turn one"})
    session_id = first.json()["session_id"]

    client.post("/api/chat", json={"message": "Turn two", "session_id": session_id})
    second_call = mock_anthropic.return_value.messages.create.call_args_list[1]
    sent_messages = second_call.kwargs["messages"]
    assert any(m["role"] == "user" and m["content"] == "Turn one" for m in sent_messages)


def test_chat_unknown_session_id_404s():
    response = client.post("/api/chat", json={"message": "Hi", "session_id": "does-not-exist"})
    assert response.status_code == 404


def test_chat_empty_session_id_400s():
    response = client.post("/api/chat", json={"message": "Hi", "session_id": ""})
    assert response.status_code == 400


@patch("app.agent.Anthropic")
def test_chat_history_ignored_when_session_id_present(mock_anthropic):
    mock_anthropic.return_value = _mock_client([_response([_text_block("ok")])])
    session_id = context.create_session()
    response = client.post("/api/chat", json={
        "message": "Hi", "session_id": session_id,
        "history": [{"role": "user", "content": "should be ignored"}],
    })
    assert response.status_code == 200
    messages = context.get_messages(session_id)
    assert all(m["content"] != "should be ignored" for m in messages)


@patch("app.agent.Anthropic")
def test_chat_legacy_history_only_seeds_a_fresh_session(mock_anthropic):
    mock_anthropic.return_value = _mock_client([_response([_text_block("ok")])])
    response = client.post("/api/chat", json={
        "message": "Hi",
        "history": [{"role": "user", "content": "earlier turn"}],
    })
    session_id = response.json()["session_id"]
    messages = context.get_messages(session_id)
    assert any(m["content"] == "earlier turn" for m in messages)


@patch("app.agent.Anthropic")
def test_chat_db_unavailable_maps_to_503(mock_anthropic):
    mock_anthropic.return_value = _mock_client([_response([_text_block("ok")])])

    def _raise(*args, **kwargs):
        raise OperationalError("database is locked")

    with patch("app.context.create_session", _raise):
        response = client.post("/api/chat", json={"message": "Hi"})
    assert response.status_code == 503


# --- skill-state persistence across /api/chat calls (closes TODOS.md gap) ---

@patch("app.agent.Anthropic")
def test_skill_scope_persists_across_chat_calls(mock_anthropic, isolate_trace_and_skills):
    _write_skill(isolate_trace_and_skills, "demo", "demo-skill", "Only does arithmetic.",
                 allowed_tools=["calculator"])
    mock_anthropic.return_value = _mock_client([
        _response([types.SimpleNamespace(
            type="tool_use", name="load_skill", input={"name": "demo-skill"}, id="t1",
            model_dump=lambda: {"type": "tool_use", "name": "load_skill",
                                 "input": {"name": "demo-skill"}, "id": "t1"})], "tool_use"),
        _response([_text_block("Loaded.")]),
        _response([_text_block("Still restricted.")]),
    ])
    first = client.post("/api/chat", json={"message": "Use the demo skill"})
    session_id = first.json()["session_id"]

    client.post("/api/chat", json={"message": "second turn", "session_id": session_id})
    second_call = mock_anthropic.return_value.messages.create.call_args_list[-1]
    tool_names = [t["name"] for t in second_call.kwargs.get("tools", [])]
    assert "calculator" in tool_names
    assert "search_workspace" not in tool_names
    # load_skill stays available so the model can switch skills later
    assert "load_skill" in tool_names


@patch("app.agent.Anthropic")
def test_different_session_unaffected_by_another_sessions_skill_scope(mock_anthropic, isolate_trace_and_skills):
    _write_skill(isolate_trace_and_skills, "demo", "demo-skill", "Only does arithmetic.",
                 allowed_tools=[])
    mock_anthropic.return_value = _mock_client([
        _response([types.SimpleNamespace(
            type="tool_use", name="load_skill", input={"name": "demo-skill"}, id="t1",
            model_dump=lambda: {"type": "tool_use", "name": "load_skill",
                                 "input": {"name": "demo-skill"}, "id": "t1"})], "tool_use"),
        _response([_text_block("Loaded.")]),
        _response([_text_block("Unrestricted here.")]),
    ])
    first = client.post("/api/chat", json={"message": "Use the demo skill"})
    restricted_session = first.json()["session_id"]

    client.post("/api/chat", json={"message": "a fresh session"})
    last_call = mock_anthropic.return_value.messages.create.call_args_list[-1]
    tool_names = [t["name"] for t in last_call.kwargs.get("tools", [])]
    assert "calculator" in tool_names
    assert "search_workspace" in tool_names


# --- chaos test: maybe_summarize partial-write never half-applies ---

def test_maybe_summarize_chaos_partial_write_never_half_applies(monkeypatch):
    monkeypatch.setattr(settings, "summarize_threshold_tokens", 1)
    session_id = context.create_session()
    context.append_message(session_id, "user", "a long enough message to exceed the token threshold")

    state = {"writes": 0}

    class _ExplodingConnection:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, *args, **kwargs):
            if sql.strip().upper().startswith("UPDATE MESSAGES"):
                state["writes"] += 1
                raise OperationalError("simulated crash mid-write")
            return self._inner.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    original_get_connection = db.get_connection
    call_count = {"n": 0}

    def _patched_get_connection():
        call_count["n"] += 1
        conn = original_get_connection()
        if call_count["n"] == 2:  # the write connection inside maybe_summarize's commit path
            return _ExplodingConnection(conn)
        return conn

    monkeypatch.setattr(db, "get_connection", _patched_get_connection)
    monkeypatch.setattr("app.context.db.get_connection", _patched_get_connection)

    client_mock = MagicMock()
    client_mock.messages.create.return_value = _response([_text_block("Summary text.")])

    with pytest.raises(OperationalError):
        context.maybe_summarize(session_id, client_mock)

    conn = original_get_connection()
    row = conn.execute("SELECT summarized FROM messages WHERE session_id = ?", (session_id,)).fetchone()
    summary_row = conn.execute("SELECT 1 FROM summaries WHERE session_id = ?", (session_id,)).fetchone()
    conn.close()
    assert row["summarized"] == 0, "messages must not be marked summarized if the update failed"
    assert summary_row is None, "the summary insert must have rolled back alongside the failed update"
