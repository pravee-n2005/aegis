from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# OpenRouter's Anthropic-Messages-compatible endpoint (verified against
# https://openrouter.ai/docs/api/api-reference/anthropic-messages/create-a-message,
# 2026-11-15): the Anthropic SDK appends /v1/messages itself, so the base
# URL must stop before /v1.
OPENROUTER_BASE_URL = "https://openrouter.ai/api"

class Settings(BaseSettings):
    openrouter_api_key: str
    # Verified live against OpenRouter's own models API (GET
    # https://openrouter.ai/api/v1/models, 2026-11-15): one of 14 models
    # whose id ends ":free" at that time, confirmed to list "tools" and
    # "tool_choice" in its supported_parameters (required - AEGIS's agent
    # loop is tool-use-first) and independently named a top-used free model
    # on OpenRouter's own free-models collection page. Free-tier model IDs
    # rotate on OpenRouter's side; if this id stops resolving, check
    # https://openrouter.ai/models?q=free for a current replacement.
    openrouter_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    max_tool_rounds: int = 4
    workspace_dir: str = "workspace"
    skills_dir: str = "skills"
    db_path: str = ".data/aegis.db"
    max_context_tokens: int = 8000
    recent_budget_tokens: int = 4000
    summarize_threshold_tokens: int = 6000
    # Minimum cosine similarity for a retrieved chunk to count as relevant
    # evidence (M4 security hardening, 2026-11-15). Measured against the
    # real sentence-transformers/all-MiniLM-L6-v2 model on 8 representative
    # query/passage pairs drawn from this codebase: relevant pairs scored
    # 0.46-0.57, unrelated pairs scored -0.08 to 0.09, borderline pairs
    # scored 0.16-0.32. 0.2 sits comfortably above the unrelated ceiling
    # with margin, below every measured relevant score, and splits the
    # inherently ambiguous borderline range. See docs/designs/
    # m4-rag-document-intelligence.md, "Post-ship correction" for the full
    # measured table. Override via SEARCH_RELEVANCE_THRESHOLD if real usage
    # shows it's miscalibrated.
    search_relevance_threshold: float = 0.2
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("openrouter_api_key")
    @classmethod
    def _require_real_openrouter_key(cls, value: str) -> str:
        """Fail at startup with a clear message, not later as an opaque
        401/403 from OpenRouter - a missing or still-placeholder key is a
        configuration mistake, not a provider outage or a wrong model id."""
        if not value or not value.strip() or value.strip() == "replace_me":
            raise ValueError(
                "OPENROUTER_API_KEY is missing or still set to the placeholder "
                "'replace_me'. Set it to a real OpenRouter API key "
                "(https://openrouter.ai/keys) in backend/.env before starting the server."
            )
        return value

settings = Settings()
