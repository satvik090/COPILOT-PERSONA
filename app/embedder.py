from __future__ import annotations

from time import perf_counter

from sentence_transformers import SentenceTransformer

from app.cache import EmbeddingCache
from app.config import EMBEDDING_MODEL
from app.observability import embedding_latency_seconds


_embedding_cache = EmbeddingCache()
_model = SentenceTransformer(EMBEDDING_MODEL)


def embed(text: str) -> list[float]:
    cached_vector = _embedding_cache.get(text)
    if cached_vector is not None:
        return cached_vector

    start_time = perf_counter()
    encoded_vector = _model.encode(text, normalize_embeddings=True)
    embedding_latency_seconds.observe(perf_counter() - start_time)

    vector = encoded_vector.tolist()
    _embedding_cache.set(text, vector)
    return vector


def embed_batch(texts: list[str]) -> list[list[float]]:
    ordered_vectors: list[list[float] | None] = [None] * len(texts)
    uncached_indexes: list[int] = []
    uncached_texts: list[str] = []

    for index, text in enumerate(texts):
        cached_vector = _embedding_cache.get(text)
        if cached_vector is not None:
            ordered_vectors[index] = cached_vector
            continue

        uncached_indexes.append(index)
        uncached_texts.append(text)

    if uncached_texts:
        start_time = perf_counter()
        encoded_vectors = _model.encode(uncached_texts, normalize_embeddings=True)
        embedding_latency_seconds.observe(perf_counter() - start_time)

        vectors = encoded_vectors.tolist()
        for index, text, vector in zip(uncached_indexes, uncached_texts, vectors):
            _embedding_cache.set(text, vector)
            ordered_vectors[index] = vector

    final_vectors: list[list[float]] = []
    for vector in ordered_vectors:
        if vector is None:
            raise RuntimeError("Embedding batch returned an incomplete result set.")
        final_vectors.append(vector)

    return final_vectors
