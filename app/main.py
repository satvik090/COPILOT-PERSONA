from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging

import chromadb
from fastapi import BackgroundTasks, FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from app.config import CHROMA_PERSIST_DIR, CORPUS_DIRS
from app.indexer import index_corpus, start_indexer, stop_indexer
from app.observability import indexer_last_run_timestamp
from app.retriever import retrieve


logger = logging.getLogger(__name__)

_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
_collection = _client.get_or_create_collection(
    name="personal_patterns",
    embedding_function=None,
)
_startup_index_task: asyncio.Task | None = None


class RetrieveRequest(BaseModel):
    context: str
    current_prompt: str


class RetrieveResponse(BaseModel):
    pattern_summary: str
    chunks: list[dict]
    confidence_scores: dict
    retrieval_time_ms: float
    cache_hit: bool


async def _run_initial_index() -> None:
    await asyncio.to_thread(index_corpus, CORPUS_DIRS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _startup_index_task

    _startup_index_task = asyncio.create_task(_run_initial_index())
    start_indexer(run_immediately=False)
    chunk_count = _collection.count()
    logger.info(
        "Startup complete. corpus_dirs=%s chunk_count=%s",
        [str(path) for path in CORPUS_DIRS],
        chunk_count,
    )

    try:
        yield
    finally:
        stop_indexer()
        if _startup_index_task is not None and not _startup_index_task.done():
            _startup_index_task.cancel()
            try:
                await _startup_index_task
            except asyncio.CancelledError:
                pass


app = FastAPI(
    title="CopilotPersona API",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_patterns(request: RetrieveRequest) -> RetrieveResponse:
    result = retrieve(request.context)
    return RetrieveResponse(**result)


@app.get("/index/status")
async def index_status() -> dict:
    last_run_timestamp = indexer_last_run_timestamp._value.get()
    last_indexed = (
        datetime.fromtimestamp(last_run_timestamp, tz=timezone.utc).isoformat()
        if last_run_timestamp > 0
        else ""
    )

    return {
        "total_chunks": _collection.count(),
        "last_indexed": last_indexed,
        "corpus_dirs": [str(path) for path in CORPUS_DIRS],
    }


@app.post("/index/trigger")
async def trigger_index(background_tasks: BackgroundTasks) -> dict[str, str]:
    background_tasks.add_task(index_corpus, CORPUS_DIRS)
    return {"status": "indexing started"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}
