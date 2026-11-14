import pytest

from app import context, db, memory
from app.config import settings


@pytest.fixture(autouse=True)
def isolate_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "aegis.db"))
    db.init_db()


def test_extract_memories_remember_that_pattern():
    session_id = context.create_session()
    result = memory.extract_memories(session_id, "Please remember that I use dark mode.")
    assert result == ["I use dark mode"]
    assert memory.list_memories(session_id)[0]["content"] == "I use dark mode"


def test_extract_memories_remember_without_that():
    session_id = context.create_session()
    result = memory.extract_memories(session_id, "remember my favorite color is blue")
    assert result == ["my favorite color is blue"]


def test_extract_memories_no_matching_pattern_returns_empty():
    session_id = context.create_session()
    result = memory.extract_memories(session_id, "What's the weather like today?")
    assert result == []
    assert memory.list_memories(session_id) == []


def test_extract_memories_malformed_input_does_not_raise():
    session_id = context.create_session()
    result = memory.extract_memories(session_id, "")
    assert result == []
    result = memory.extract_memories(session_id, "remember")
    assert result == []


def test_extract_memories_unknown_session_does_not_raise():
    result = memory.extract_memories("does-not-exist", "remember that this should not persist")
    assert result == []


def test_list_memories_empty_for_fresh_session():
    session_id = context.create_session()
    assert memory.list_memories(session_id) == []


def test_delete_memory_removes_row():
    session_id = context.create_session()
    memory.extract_memories(session_id, "remember that I like tea")
    [m] = memory.list_memories(session_id)
    memory.delete_memory(session_id, m["id"])
    assert memory.list_memories(session_id) == []


def test_delete_memory_unknown_id_raises_not_found():
    session_id = context.create_session()
    with pytest.raises(memory.MemoryNotFoundError):
        memory.delete_memory(session_id, 999)


def test_delete_memory_wrong_session_is_404_not_success():
    session_a = context.create_session()
    session_b = context.create_session()
    memory.extract_memories(session_a, "remember that A's secret is foo")
    [m] = memory.list_memories(session_a)

    with pytest.raises(memory.MemoryNotFoundError):
        memory.delete_memory(session_b, m["id"])

    # the memory must still exist under its real session
    assert memory.list_memories(session_a) == [m]


def test_deleted_memory_never_reappears_in_build_context():
    session_id = context.create_session()
    memory.extract_memories(session_id, "remember that I like tea")
    [m] = memory.list_memories(session_id)
    memory.delete_memory(session_id, m["id"])

    system_prompt_addition, _ = context.build_context(session_id)
    assert "I like tea" not in system_prompt_addition


def test_clear_memories_removes_all_for_session_only():
    session_a = context.create_session()
    session_b = context.create_session()
    memory.extract_memories(session_a, "remember that A's secret is foo")
    memory.extract_memories(session_b, "remember that B's secret is bar")

    removed = memory.clear_memories(session_a)
    assert removed == 1
    assert memory.list_memories(session_a) == []
    assert len(memory.list_memories(session_b)) == 1
