"""Centralized configuration, loaded from environment variables / .env."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"

    data_dir: str = "./data"
    upload_dir: str = "./data/uploads"
    chroma_dir: str = "./data/chroma"

    top_k: int = 6
    chunk_size: int = 800
    chunk_overlap: int = 150

    embedding_model: str = "all-MiniLM-L6-v2"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.upload_dir, self.chroma_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()