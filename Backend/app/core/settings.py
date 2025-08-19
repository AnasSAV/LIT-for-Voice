from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
    )
    REDIS_URL: str = "redis://localhost:6379/0"
    SESSION_COOKIE_NAME: str = "sid"
    SESSION_TTL_SECONDS: int = 24 * 60 * 60
    DATASET_CACHE_TTL_SECONDS: int = 24 * 60 * 60
    COOKIE_SECURE: bool = False
    COOKIE_SAMESITE: str = "lax"
    COOKIE_DOMAIN: str | None = None

settings = Settings()