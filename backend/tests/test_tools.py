import pytest
from app.tools import safe_calculate, search_workspace

def test_safe_calculate_basic_math():
    assert safe_calculate("(12 + 8) * 2") == 40

def test_safe_calculate_rejects_code():
    with pytest.raises((ValueError, TypeError)):
        safe_calculate("__import__('os').system('id')")

def test_safe_calculate_rejects_large_exponent():
    with pytest.raises(ValueError):
        safe_calculate("2 ** 99")

def test_search_workspace_finds_text(tmp_path):
    (tmp_path / "notes.md").write_text(
        "# RAG\nEmbeddings retrieve relevant chunks.", encoding="utf-8")
    result = search_workspace("embeddings", str(tmp_path))
    assert result[0]["file"] == "notes.md"

def test_search_workspace_empty_query(tmp_path):
    assert search_workspace("   ", str(tmp_path)) == []
