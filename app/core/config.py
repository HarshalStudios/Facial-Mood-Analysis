from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    app_name: str = "Facial Mood Analysis"
    environment: str = "development"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: list[str] = ["*"]

    class Config:
        env_prefix = "FMA_"
        env_file = ".env"

@lru_cache
def get_settings():
    return Settings()
