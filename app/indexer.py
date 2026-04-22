from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path
import time

from apscheduler.schedulers.asyncio import AsyncIOScheduler
import chromadb

from app import classifier, embedder, extractor
from app.config import CHROMA_PERSIST_DIR, CORPUS_DIRS, INDEXER_SCHEDULE_HOURS
from app.observability import index_total_chunks, indexer_last_run_timestamp


logger = logging.getLogger(__name__)

_excluded_directories = {
    "__pycache__",
    ".git",
    "venv",
    "node_modules",
    "site-packages",
    "migrations",
    "dist",
    "build",
}

_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
_collection = _client.get_or_create_collection(
    name="personal_patterns",
    embedding_function=None,
)
_scheduler: AsyncIOScheduler | None = None
_has_started_once = False


def index_file(file_path: Path) -> int:
    functions = extractor.extract_functions(file_path)
    featured_functions = [extractor.extract_features(func) for func in functions]

    if featured_functions:
        file_pattern_summary = classifier.classify_patterns(featured_functions)
        if file_pattern_summary["summary"]:
            logger.debug(
                "Pattern summary for %s: %s",
                file_path,
                file_pattern_summary["summary"],
            )

    indexed_count = 0
    indexed_at = datetime.now(timezone.utc).isoformat()
    project_name = _get_project_name(file_path)

    for func in featured_functions:
        text_to_embed = (
            func["signature"]
            + "\n"
            + (func["docstring"] or "")
            + "\n"
            + func["full_source"]
        )
        vector = embedder.embed(text_to_embed)
        doc_id = _build_doc_id(file_path, func["name"], func["line_count"])

        metadata = {
            "file_path": func["file_path"],
            "name": func["name"],
            "signature": func["signature"],
            "naming_convention": func["naming_convention"],
            "error_handling": func["error_handling"],
            "docstring_format": func["docstring_format"],
            "annotation_density": func["annotation_density"],
            "project": project_name,
            "indexed_at": indexed_at,
        }

        _collection.upsert(
            ids=[doc_id],
            embeddings=[vector],
            documents=[func["full_source"]],
            metadatas=[metadata],
        )
        indexed_count += 1

    return indexed_count


def index_corpus(dirs: list[Path]) -> int:
    total_chunks_indexed = 0
    total_files_processed = 0

    for corpus_dir in dirs:
        for file_path in corpus_dir.rglob("*.py"):
            if _should_skip_path(file_path):
                continue

            total_files_processed += 1
            total_chunks_indexed += index_file(file_path)

    index_total_chunks.set(total_chunks_indexed)
    indexer_last_run_timestamp.set(time.time())
    logger.info(
        "Indexed %s files and %s chunks.",
        total_files_processed,
        total_chunks_indexed,
    )
    return total_chunks_indexed


def start_indexer(run_immediately: bool = True) -> AsyncIOScheduler:
    global _scheduler
    global _has_started_once

    if _scheduler is None:
        _scheduler = AsyncIOScheduler()

    if run_immediately and not _has_started_once:
        index_corpus(CORPUS_DIRS)
        _has_started_once = True
    elif not run_immediately:
        _has_started_once = True

    _scheduler.add_job(
        index_corpus,
        trigger="interval",
        hours=INDEXER_SCHEDULE_HOURS,
        args=[CORPUS_DIRS],
        id="personal-pattern-indexer",
        replace_existing=True,
    )

    if not _scheduler.running:
        _scheduler.start()

    return _scheduler


def stop_indexer() -> None:
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown(wait=False)


def _should_skip_path(file_path: Path) -> bool:
    return any(part in _excluded_directories for part in file_path.parts)


def _build_doc_id(file_path: Path, function_name: str, line_count: int) -> str:
    value = f"{file_path}{function_name}{line_count}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _get_project_name(file_path: Path) -> str:
    if len(file_path.parts) >= 3:
        return file_path.parts[-3]
    return "unknown"
