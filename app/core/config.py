from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "development")
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
    allow_cloud_llm: bool = os.getenv("ALLOW_CLOUD_LLM", "false").lower() == "true"
    vector_backend: str = os.getenv("VECTOR_BACKEND", "chroma")
    chroma_dir: Path = ROOT_DIR / os.getenv("CHROMA_DIR", "data/vectorstore/chroma")
    sqlite_path: Path = ROOT_DIR / os.getenv("SQLITE_PATH", "sessions/rca_copilot.sqlite")
    max_refinements: int = int(os.getenv("MAX_REFINEMENTS", "2"))
    grounding_threshold: float = float(os.getenv("GROUNDING_THRESHOLD", "0.72"))
    faithfulness_threshold: float = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.70"))
    skills_path: Path = ROOT_DIR / "skills.md"
    manuals_dir: Path = ROOT_DIR / "data/manuals"
    raw_data_dir: Path = ROOT_DIR / "data/raw"
    exports_dir: Path = ROOT_DIR / "exports"


settings = Settings()

