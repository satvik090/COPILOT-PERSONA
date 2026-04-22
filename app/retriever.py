from __future__ import annotations

from time import perf_counter

import chromadb

from app import classifier, embedder
from app.cache import RetrievalCache, get_context_hash
from app.config import CHROMA_PERSIST_DIR, RETRIEVAL_TOP_K
from app.observability import (
    pattern_confidence_score,
    retrieval_cache_hits,
    retrieval_cache_misses,
    retrieval_latency_seconds,
)


_cache = RetrievalCache()
_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
_collection = _client.get_or_create_collection(
    name="personal_patterns",
    embedding_function=None,
)


def retrieve(context: str) -> dict:
    context_hash = get_context_hash(context)
    cached_result = _cache.get(context_hash)

    if cached_result is not None:
        retrieval_cache_hits.inc()
        return {**cached_result, "cache_hit": True}

    retrieval_cache_misses.inc()
    start_time = perf_counter()
    vector = embedder.embed(context)
    query_result = _collection.query(
        query_embeddings=[vector],
        n_results=RETRIEVAL_TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    documents = query_result.get("documents", [[]])
    metadatas = query_result.get("metadatas", [[]])
    distances = query_result.get("distances", [[]])

    chunks: list[dict] = []
    for metadata, document, distance in zip(
        metadatas[0] if metadatas else [],
        documents[0] if documents else [],
        distances[0] if distances else [],
    ):
        chunk = {
            **(metadata or {}),
            "document": document,
            "distance": distance,
        }
        chunks.append(chunk)

    classification = classifier.classify_patterns(chunks)
    retrieval_time_ms = (perf_counter() - start_time) * 1000

    result = {
        "pattern_summary": classification["summary"],
        "chunks": chunks,
        "confidence_scores": classification["confidence_scores"],
        "retrieval_time_ms": retrieval_time_ms,
        "cache_hit": False,
    }

    _cache.set(
        context_hash,
        {
            "pattern_summary": result["pattern_summary"],
            "chunks": result["chunks"],
            "confidence_scores": result["confidence_scores"],
            "retrieval_time_ms": result["retrieval_time_ms"],
            "cache_hit": False,
        },
    )
    retrieval_latency_seconds.observe(retrieval_time_ms / 1000)

    for score in classification["confidence_scores"].values():
        pattern_confidence_score.observe(score)

    return result
