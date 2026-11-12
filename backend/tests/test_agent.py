import os
import types
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

from app import tracing  # noqa: E402
from app.agent import ask_aegis  # noqa: E402


@pytest.fixture(autouse=True)
def isolate_trace_file(tmp_path, monkeypatch):
    """Keep test runs from appending to the real backend/workspace/.traces/traces.jsonl."""
    monkeypatch.setattr(tracing, "TRACE_PATH", tmp_path / "traces.jsonl")


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
