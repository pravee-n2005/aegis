import os
import types
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from app import tracing  # noqa: E402
from app.agent import ask_aegis, TOOLS  # noqa: E402
from app.config import settings  # noqa: E402


@pytest.fixture(autouse=True)
def isolate_trace_file(tmp_path, monkeypatch):
    """Keep test runs from appending to the real backend/workspace/.traces/traces.jsonl."""
    monkeypatch.setattr(tracing, "TRACE_PATH", tmp_path / "traces.jsonl")


@pytest.fixture(autouse=True)
def isolate_skills_dir(tmp_path, monkeypatch):
    """Point skills_dir at an empty tmp directory by default, so every existing
    test exercises plain M1 behavior unless it explicitly writes a skill fixture
    into this same directory. Mirrors isolate_trace_file's pattern."""
    monkeypatch.setattr(settings, "skills_dir", str(tmp_path / "skills"))
    return tmp_path / "skills"


def _write_skill(skills_dir, dirname, name, description, allowed_tools=None, body="Follow these instructions."):
    skill_dir = skills_dir / dirname
    skill_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = f"name: {name}\ndescription: {description}"
    if allowed_tools is not None:
        tools_yaml = "\n".join(f"  - {t}" for t in allowed_tools) or "[]"
        frontmatter += f"\nallowed-tools:\n{tools_yaml}" if allowed_tools else "\nallowed-tools: []"
    (skill_dir / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n{body}", encoding="utf-8")


def _text_block(text):
    return types.SimpleNamespace(type="text", text=text)


def _tool_use_block(name, input_, block_id="tool_1"):
    return types.SimpleNamespace(type="tool_use", name=name, input=input_, id=block_id)


def _response(content, stop_reason):
    return types.SimpleNamespace(content=content, stop_reason=stop_reason)


def _mock_client(responses):
    client = MagicMock()
    client.messages.create.side_effect = responses
    return client


def _tool_names(call):
    return [t["name"] for t in call.kwargs.get("tools", [])]


@patch("app.agent.Anthropic")
def test_normal_response(mock_anthropic):
    mock_anthropic.return_value = _mock_client([
        _response([_text_block("Hello there.")], "end_turn"),
    ])
    result = ask_aegis("Hi")
    assert result["answer"] == "Hello there."
    assert result["trace"] == []
    assert result["stop_reason"] == "end_turn"


@patch("app.agent.Anthropic")
def test_successful_tool_execution(mock_anthropic):
    mock_anthropic.return_value = _mock_client([
        _response([_tool_use_block("calculator", {"expression": "2 + 2"})], "tool_use"),
        _response([_text_block("The result is 4.")], "end_turn"),
    ])
    result = ask_aegis("What is 2 + 2?")
    assert result["answer"] == "The result is 4."
    assert result["stop_reason"] == "end_turn"
    assert len(result["trace"]) == 1
    assert result["trace"][0]["tool"] == "calculator"
    assert result["trace"][0]["result"] == {"result": 4}


@patch("app.agent.Anthropic")
def test_tool_execution_failure(mock_anthropic):
    mock_anthropic.return_value = _mock_client([
        _response([_tool_use_block("calculator", {"expression": "__import__('os')"})], "tool_use"),
        _response([_text_block("I could not evaluate that expression.")], "end_turn"),
    ])
    result = ask_aegis("Run some code for me")
    assert len(result["trace"]) == 1
    assert "error" in result["trace"][0]["result"]
    assert result["answer"] == "I could not evaluate that expression."


@patch("app.agent.Anthropic")
def test_max_tool_round_exhaustion(mock_anthropic):
    tool_round = _response([_tool_use_block("calculator", {"expression": "1 + 1"})], "tool_use")
    mock_anthropic.return_value = _mock_client([tool_round, tool_round, tool_round, tool_round])
    result = ask_aegis("Keep calculating forever")
    assert result["stop_reason"] == "tool_round_limit"
    assert result["answer"].startswith("Tool-use limit reached")
    assert len(result["trace"]) == 4


# --- M2: skills ---

@patch("app.agent.Anthropic")
def test_m1_regression_no_skills_directory(mock_anthropic):
    """With no skills directory, tools= and system= passed to the API must be
    byte-for-byte what M1 always sent — proves the feature is fully inert
    when absent."""
    mock_client = _mock_client([_response([_text_block("Hello there.")], "end_turn")])
    mock_anthropic.return_value = mock_client
    from app.agent import SYSTEM_PROMPT
    result = ask_aegis("Hi")
    assert result["answer"] == "Hello there."
    call = mock_client.messages.create.call_args
    assert call.kwargs["tools"] == TOOLS
    assert call.kwargs["system"] == SYSTEM_PROMPT


@patch("app.agent.Anthropic")
def test_skill_metadata_injected_when_skill_exists(mock_anthropic, isolate_skills_dir):
    _write_skill(isolate_skills_dir, "demo", "demo-skill", "A demo skill for testing.")
    mock_client = _mock_client([_response([_text_block("Hi.")], "end_turn")])
    mock_anthropic.return_value = mock_client
    ask_aegis("Hello")
    call = mock_client.messages.create.call_args
    assert "demo-skill" in call.kwargs["system"]
    assert "A demo skill for testing." in call.kwargs["system"]
    assert "load_skill" in _tool_names(call)


@patch("app.agent.Anthropic")
def test_skill_metadata_never_includes_body(mock_anthropic, isolate_skills_dir):
    _write_skill(isolate_skills_dir, "demo", "demo-skill", "A demo skill.", body="TOP SECRET INSTRUCTIONS")
    mock_client = _mock_client([_response([_text_block("Hi.")], "end_turn")])
    mock_anthropic.return_value = mock_client
    ask_aegis("Hello")
    call = mock_client.messages.create.call_args
    assert "TOP SECRET INSTRUCTIONS" not in call.kwargs["system"]


@patch("app.agent.Anthropic")
def test_load_skill_scopes_tools_for_next_round(mock_anthropic, isolate_skills_dir):
    _write_skill(isolate_skills_dir, "demo", "demo-skill", "Only does arithmetic.",
                 allowed_tools=["calculator"])
    mock_client = _mock_client([
        _response([_tool_use_block("load_skill", {"name": "demo-skill"})], "tool_use"),
        _response([_text_block("Done.")], "end_turn"),
    ])
    mock_anthropic.return_value = mock_client
    result = ask_aegis("Use the demo skill")
    assert result["answer"] == "Done."
    first_call, second_call = mock_client.messages.create.call_args_list
    assert sorted(_tool_names(first_call)) == ["calculator", "load_skill", "search_workspace"]
    assert _tool_names(second_call) == ["calculator"]
    assert result["trace"][0]["tool"] == "load_skill"
    assert result["trace"][0]["result"] == {"instructions": "Follow these instructions."}


@patch("app.agent.Anthropic")
def test_load_skill_empty_allowlist_denies_all_tools(mock_anthropic, isolate_skills_dir):
    _write_skill(isolate_skills_dir, "demo", "demo-skill", "Does nothing tool-wise.",
                 allowed_tools=[])
    mock_client = _mock_client([
        _response([_tool_use_block("load_skill", {"name": "demo-skill"})], "tool_use"),
        _response([_text_block("Done.")], "end_turn"),
    ])
    mock_anthropic.return_value = mock_client
    ask_aegis("Use the demo skill")
    _, second_call = mock_client.messages.create.call_args_list
    assert _tool_names(second_call) == []
    # omitted entirely, not an explicit [] — some SDKs/APIs treat them differently
    assert "tools" not in second_call.kwargs


@patch("app.agent.Anthropic")
def test_load_skill_name_mismatch_returns_error(mock_anthropic, isolate_skills_dir):
    _write_skill(isolate_skills_dir, "demo", "demo-skill", "A demo skill.")
    mock_client = _mock_client([
        _response([_tool_use_block("load_skill", {"name": "nonexistent"})], "tool_use"),
        _response([_text_block("Could not find that skill.")], "end_turn"),
    ])
    mock_anthropic.return_value = mock_client
    result = ask_aegis("Load a skill that doesn't exist")
    assert "error" in result["trace"][0]["result"]
    _, second_call = mock_client.messages.create.call_args_list
    # no skill was loaded, so load_skill is still offered on the next round
    assert "load_skill" in _tool_names(second_call)


@patch("app.agent.Anthropic")
def test_second_load_skill_call_in_same_round_rejected(mock_anthropic, isolate_skills_dir):
    _write_skill(isolate_skills_dir, "a", "skill-a", "First skill.")
    _write_skill(isolate_skills_dir, "b", "skill-b", "Second skill.")
    mock_client = _mock_client([
        _response([
            _tool_use_block("load_skill", {"name": "skill-a"}, block_id="t1"),
            _tool_use_block("load_skill", {"name": "skill-b"}, block_id="t2"),
        ], "tool_use"),
        _response([_text_block("Done.")], "end_turn"),
    ])
    mock_anthropic.return_value = mock_client
    result = ask_aegis("Load two skills")
    first, second = result["trace"]
    assert first["result"] == {"instructions": "Follow these instructions."}
    assert "error" in second["result"]
    assert "already loaded" in second["result"]["error"].lower()


@patch("app.agent.Anthropic")
def test_trace_includes_skill_lifecycle_events(mock_anthropic, isolate_skills_dir):
    _write_skill(isolate_skills_dir, "demo", "demo-skill", "A demo skill.", allowed_tools=["calculator"])
    mock_client = _mock_client([
        _response([_tool_use_block("load_skill", {"name": "demo-skill"})], "tool_use"),
        _response([_text_block("Done.")], "end_turn"),
    ])
    mock_anthropic.return_value = mock_client
    ask_aegis("Use the demo skill")
    import json
    records = [json.loads(line) for line in tracing.TRACE_PATH.read_text().splitlines()]
    events = {r["event"] for r in records}
    assert "skill_discovered" in events
    assert "skill_loaded" in events
    assert "skill_tool_scoping" in events
    scoping = [r for r in records if r["event"] == "skill_tool_scoping"][0]
    assert scoping["reason"] == "TOOL_DENIED"
    assert "search_workspace" in scoping["denied_tools"]
