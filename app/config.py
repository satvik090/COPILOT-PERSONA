from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH if ENV_PATH.exists() else BASE_DIR / ".env.example")


def _parse_corpus_dirs(raw_value: str) -> list[Path]:
    return [Path(item.strip()) for item in raw_value.split(",") if item.strip()]


CORPUS_DIRS: list[Path] = _parse_corpus_dirs(os.getenv("CORPUS_DIRS", "./my_projects"))
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "your_key_here")
CHROMA_PERSIST_DIR: Path = Path(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"))
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "10"))
PATTERN_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("PATTERN_CONFIDENCE_THRESHOLD", "0.7")
)
CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "30"))
MAX_INJECTION_TOKENS: int = int(os.getenv("MAX_INJECTION_TOKENS", "150"))
INDEXER_SCHEDULE_HOURS: int = int(os.getenv("INDEXER_SCHEDULE_HOURS", "6"))

SUPPORTED_EXTENSIONS = [".py"]
MIN_FUNCTION_LINES = 5
