from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-4-6"
    max_tool_rounds: int = 4
    workspace_dir: str = "workspace"
    skills_dir: str = "skills"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
