from prometheus_client import Counter, Gauge, Histogram, make_asgi_app


index_total_chunks = Gauge(
    "index_total_chunks",
    "Total number of chunks currently stored by the indexer.",
)

indexer_last_run_timestamp = Gauge(
    "indexer_last_run_timestamp",
    "Unix timestamp of the indexer's most recent completed run.",
)

retrieval_latency_seconds = Histogram(
    "retrieval_latency_seconds",
    "Latency for retrieval operations in seconds.",
    buckets=(0.01, 0.05, 0.1, 0.15, 0.2, 0.5),
)

retrieval_cache_hits = Counter(
    "retrieval_cache_hits",
    "Number of retrieval cache hits.",
)

retrieval_cache_misses = Counter(
    "retrieval_cache_misses",
    "Number of retrieval cache misses.",
)

pattern_confidence_score = Histogram(
    "pattern_confidence_score",
    "Confidence scores assigned by the pattern classifier.",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

embedding_latency_seconds = Histogram(
    "embedding_latency_seconds",
    "Latency for embedding generation in seconds.",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1),
)


def metrics_app():
    return make_asgi_app()
