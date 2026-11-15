from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-4-6"
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

settings = Settings()
