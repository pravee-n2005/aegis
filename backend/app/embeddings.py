"""Local embedding backend for M4 document retrieval.

A single swappable function, embed_texts(), stands between documents.py's
ingestion path and retrieval.py's query path and whichever embedding
provider actually runs - today a local sentence-transformers model, later
possibly a hosted API behind the same signature (docs/designs/
m4-rag-document-intelligence.md, "Dependency choices and trade-offs").

The model is loaded once per process as a module-level lazy singleton -
never per-request, never per-chunk - because loading it is itself a slow,
occasionally-failing operation (first use downloads ~90MB from HuggingFace
Hub). Pinned to an explicit revision so a model swap on HuggingFace's side
can never silently change embedding behavior underneath an already-stored
corpus.

Tests never hit this real backend: they inject a fake embedder via
monkeypatch (the same pattern this codebase already uses for
client.messages.create), so no test run needs network access, torch, or a
real model.
"""
import threading

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

_model = None
_model_lock = threading.Lock()


class EmbeddingUnavailableError(Exception):
    """Raised when the embedding backend can't be loaded or can't run -
    missing dependency, corrupted HuggingFace cache, no network on first
    use, or an inference failure. Callers map this to a 503 (a service-
    availability failure), distinct from a 422 input-validation failure."""


def _load_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingUnavailableError(f"embedding dependency not installed: {exc}") from exc
        try:
            _model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION)
        except Exception as exc:
            raise EmbeddingUnavailableError(f"embedding model failed to load: {exc}") from exc
        return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts in one call - callers must batch (e.g. all of
    a document's chunks in one call), never call this per-chunk, so the
    model-loading/inference overhead is amortized once per document, not
    once per chunk."""
    if not texts:
        return []
    model = _load_model()
    try:
        vectors = model.encode(texts)
    except Exception as exc:
        raise EmbeddingUnavailableError(f"embedding inference failed: {exc}") from exc
    return [list(float(x) for x in vector) for vector in vectors]
