"""Tests for Phase 4 vector store."""

from unittest.mock import MagicMock

import pytest

from src.config.models import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME
from src.config.vectordb import COLLECTION_NAME
from src.ingestion.embedder import EmbeddedChunk
from src.ingestion.store import (
    VectorStoreError,
    get_collection_ingestion_timestamp,
    query_similar,
    run_vector_store,
    upsert_embeddings,
)


@pytest.fixture
def sample_embeddings() -> list[EmbeddedChunk]:
    return [
        EmbeddedChunk(
            chunk_id="test-fund-001",
            text="Fund: Test Fund\nKey fact: Expense ratio: 1.19%",
            fund_name="Test Fund",
            source_url="https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
            ingestion_timestamp="2026-08-30T06:21:03Z",
            embedding=[0.1] * EMBEDDING_DIMENSION,
            embedding_model=EMBEDDING_MODEL_NAME,
            embedding_dim=EMBEDDING_DIMENSION,
        ),
        EmbeddedChunk(
            chunk_id="test-fund-002",
            text="Fund: Test Fund\nKey fact: Lock-in period: 3 year(s)",
            fund_name="Test Fund",
            source_url="https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
            ingestion_timestamp="2026-08-30T06:21:03Z",
            embedding=[0.2] * EMBEDDING_DIMENSION,
            embedding_model=EMBEDDING_MODEL_NAME,
            embedding_dim=EMBEDDING_DIMENSION,
        ),
    ]


def test_upsert_embeddings_stores_records(tmp_path, sample_embeddings):
    collection = upsert_embeddings(
        sample_embeddings,
        persist_path=tmp_path / "vectordb",
        collection_name="test_collection",
    )
    assert collection.count() == 2
    assert get_collection_ingestion_timestamp(
        tmp_path / "vectordb", "test_collection"
    ) == "2026-08-30T06:21:03Z"


def test_upsert_embeddings_replaces_collection_on_rerun(tmp_path, sample_embeddings):
    path = tmp_path / "vectordb"
    upsert_embeddings(sample_embeddings, persist_path=path, collection_name="test_collection")
    upsert_embeddings(sample_embeddings[:1], persist_path=path, collection_name="test_collection")

    from src.ingestion.store import get_collection

    collection = get_collection(path, "test_collection")
    assert collection.count() == 1


def test_query_similar_returns_results(tmp_path, sample_embeddings):
    path = tmp_path / "vectordb"
    upsert_embeddings(sample_embeddings, persist_path=path, collection_name="test_collection")

    results = query_similar(
        query_embedding=[0.1] * EMBEDDING_DIMENSION,
        top_k=1,
        persist_path=path,
        collection_name="test_collection",
    )
    assert len(results) == 1
    assert results[0].chunk_id == "test-fund-001"
    assert results[0].fund_name == "Test Fund"
    assert results[0].source_url.startswith("https://groww.in/")


def test_upsert_rejects_empty_embeddings():
    with pytest.raises(VectorStoreError, match="empty embedding list"):
        upsert_embeddings([])


def test_run_vector_store_integration(monkeypatch, tmp_path, sample_embeddings):
    mock_collection = MagicMock()
    mock_collection.count.return_value = len(sample_embeddings)
    mock_collection.metadata = {"ingestion_timestamp": "2026-08-30T06:21:03Z"}

    monkeypatch.setattr(
        "src.ingestion.store.load_embeddings",
        lambda path=None: sample_embeddings,
    )
    monkeypatch.setattr(
        "src.ingestion.store.upsert_embeddings",
        lambda embeddings, persist_path=None, collection_name=COLLECTION_NAME: mock_collection,
    )
    monkeypatch.setattr(
        "src.ingestion.store.get_collection_ingestion_timestamp",
        lambda persist_path=None, collection_name=COLLECTION_NAME: "2026-08-30T06:21:03Z",
    )
    monkeypatch.setattr(
        "src.ingestion.store.EMBEDDING_MANIFEST_PATH",
        tmp_path / "embedding_manifest.json",
    )
    (tmp_path / "embedding_manifest.json").write_text(
        '{"status": "success", "embedding_timestamp": "2026-08-30T06:45:18Z", '
        '"embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "embedding_dim": 384}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.ingestion.store.VECTOR_STORE_MANIFEST_PATH",
        tmp_path / "vector_store_manifest.json",
    )

    manifest = run_vector_store(persist_path=tmp_path / "vectordb")
    assert manifest.status == "success"
    assert manifest.chunks_stored == 2
    assert manifest.collection_name == COLLECTION_NAME
    assert manifest.ingestion_timestamp == "2026-08-30T06:21:03Z"
