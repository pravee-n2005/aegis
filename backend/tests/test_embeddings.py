import sys

import pytest

from app import embeddings


@pytest.fixture(autouse=True)
def reset_model_singleton():
    """Every test starts from an unloaded singleton and leaves it unloaded,
    so no test's stub model leaks into another test."""
    embeddings._model = None
    yield
    embeddings._model = None


def test_embed_texts_empty_list_returns_empty_without_loading_model():
    assert embeddings.embed_texts([]) == []
    assert embeddings._model is None


def test_load_model_wraps_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(embeddings.EmbeddingUnavailableError):
        embeddings.embed_texts(["hello"])


def test_load_model_wraps_construction_failure(monkeypatch):
    class _FakeModule:
        class SentenceTransformer:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("corrupted HuggingFace cache")

    monkeypatch.setitem(sys.modules, "sentence_transformers", _FakeModule)

    with pytest.raises(embeddings.EmbeddingUnavailableError):
        embeddings.embed_texts(["hello"])


def test_embed_texts_wraps_inference_failure(monkeypatch):
    class _FakeModel:
        def encode(self, texts):
            raise RuntimeError("inference blew up")

    monkeypatch.setattr(embeddings, "_model", _FakeModel())

    with pytest.raises(embeddings.EmbeddingUnavailableError):
        embeddings.embed_texts(["hello"])


def test_embed_texts_happy_path_with_stub_model(monkeypatch):
    class _FakeModel:
        def encode(self, texts):
            return [[float(len(t)), 0.0] for t in texts]

    monkeypatch.setattr(embeddings, "_model", _FakeModel())

    vectors = embeddings.embed_texts(["hi", "hello"])

    assert vectors == [[2.0, 0.0], [5.0, 0.0]]
