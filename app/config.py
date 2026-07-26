from functools import lru_cache
from typing import Literal

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AI Pets Agent API"
    app_env: Literal["development", "test", "production"] = "development"
    database_url: str = "sqlite+aiosqlite:///./agent.db"

    model_provider: Literal["mock", "openai_compatible"] = "mock"
    model_base_url: str = "https://example.invalid/v1"
    model_api_key: SecretStr = SecretStr("")
    model_name: str = "mock-model"
    model_timeout_seconds: float = 30
    model_max_retries: int = 2
    model_input_price_per_million: float = 0
    model_output_price_per_million: float = 0

    # Matching funnel parameters. These are experiment knobs, not final values.
    matching_min_shared_memory_keys: int = 2
    matching_min_compatibility_score: float = 0.2
    matching_dialogue_rounds: int = 2
    matching_judge_min_confidence: float = 0.6

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_model_credentials(self) -> "Settings":
        if (
            self.model_provider == "openai_compatible"
            and not self.model_api_key.get_secret_value()
        ):
            raise ValueError("MODEL_API_KEY is required for openai_compatible provider")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
