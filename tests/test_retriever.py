from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import chromadb
import pytest

from app.cache import RetrievalCache


def _install_fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.ModuleType("sentence_transformers")

    class FakeSentenceTransformer:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def encode(self, texts, normalize_embeddings: bool = True):
            if isinstance(texts, str):
                return FakeVector([0.0] * 384)
            return FakeMatrix([[0.0] * 384 for _ in texts])

    class FakeVector(list):
        def tolist(self) -> list[float]:
            return list(self)

    class FakeMatrix(list):
        def tolist(self) -> list[list[float]]:
            return [list(item) for item in self]

    fake_module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)


@pytest.fixture
def retriever_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _install_fake_sentence_transformers(monkeypatch)

    sys.modules.pop("app.embedder", None)
    sys.modules.pop("app.retriever", None)

    import app.config as config

    monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", tmp_path)

    retriever = importlib.import_module("app.retriever")
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.get_or_create_collection(
        name="personal_patterns",
        embedding_function=None,
    )

    monkeypatch.setattr(retriever, "CHROMA_PERSIST_DIR", tmp_path)
    monkeypatch.setattr(retriever, "_client", client)
    monkeypatch.setattr(retriever, "_collection", collection)
    monkeypatch.setattr(retriever, "_cache", RetrievalCache())
    monkeypatch.setattr(retriever.embedder, "embed", lambda text: [0.0] * 384)

    return retriever, collection


def _insert_synthetic_chunks(collection) -> None:
    collection.upsert(
        ids=[f"chunk-{index}" for index in range(5)],
        embeddings=[[0.0] * 384 for _ in range(5)],
        documents=[f"def example_{index}():\n    return {index}" for index in range(5)],
        metadatas=[
            {
                "file_path": f"/tmp/project/module_{index}.py",
                "name": f"example_{index}",
                "signature": f"def example_{index}()",
                "naming_convention": "snake_case",
                "error_handling": "return_tuple",
                "docstring_format": "google",
                "annotation_density": "full",
                "project": "project",
                "indexed_at": "2026-01-01T00:00:00+00:00",
            }
            for index in range(5)
        ],
    )


def test_retrieve_returns_dict_with_required_keys(retriever_module) -> None:
    retriever, collection = retriever_module
    _insert_synthetic_chunks(collection)

    result = retriever.retrieve("calculate totals for invoices")

    assert "pattern_summary" in result
    assert "chunks" in result
    assert "confidence_scores" in result
    assert "retrieval_time_ms" in result
    assert isinstance(result["chunks"], list)
    assert result["retrieval_time_ms"] >= 0.0


def test_retrieval_cache_hit_on_second_call(retriever_module) -> None:
    retriever, collection = retriever_module
    _insert_synthetic_chunks(collection)

    first_result = retriever.retrieve("build billing summary")
    second_result = retriever.retrieve("build billing summary")

    assert first_result["cache_hit"] is False
    assert second_result["cache_hit"] is True


def test_retrieval_returns_top_k_or_fewer_chunks(retriever_module) -> None:
    retriever, collection = retriever_module
    _insert_synthetic_chunks(collection)

    result = retriever.retrieve("rank personal patterns")

    assert len(result["chunks"]) <= 5


def test_empty_index_returns_empty_summary(retriever_module) -> None:
    retriever, _ = retriever_module

    result = retriever.retrieve("context with no indexed chunks")

    assert result["chunks"] == []
    assert result["pattern_summary"] == ""
