from __future__ import annotations

import hashlib
import time

from app.config import CACHE_TTL_SECONDS


def _short_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def get_context_hash(context: str) -> str:
    return _short_sha256(context)


class EmbeddingCache:
    def __init__(self) -> None:
        self._store: dict[str, list[float]] = {}

    def get(self, text: str) -> list[float] | None:
        key = _short_sha256(text)
        return self._store.get(key)

    def set(self, text: str, vector: list[float]) -> None:
        key = _short_sha256(text)
        self._store[key] = vector

    def size(self) -> int:
        return len(self._store)


class RetrievalCache:
    def __init__(self) -> None:
        self._store: dict[str, tuple[float, dict]] = {}

    def get(self, context_hash: str) -> dict | None:
        cached = self._store.get(context_hash)
        if cached is None:
            return None

        expires_at, result = cached
        if expires_at <= time.time():
            self._store.pop(context_hash, None)
            return None

        return result

    def set(self, context_hash: str, result: dict) -> None:
        expires_at = time.time() + CACHE_TTL_SECONDS
        self._store[context_hash] = (expires_at, result)

    def invalidate_all(self) -> None:
        self._store.clear()
