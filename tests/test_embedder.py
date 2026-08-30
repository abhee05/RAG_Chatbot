"""Tests for Phase 3 embedding."""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config.models import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME
from src.ingestion.chunker import Chunk
from src.ingestion.embedder import (
    EmbeddingError,
    Embedder,
    embed_chunks,
    load_chunks,
    run_embedding,
)


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="test-fund-001",
            text="Fund: Test Fund\nKey fact: Expense ratio: 1.19%",
            fund_name="Test Fund",
            source_url="https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
            ingestion_timestamp="2026-08-30T06:21:03Z",
        ),
        Chunk(
            chunk_id="test-fund-002",
            text="Fund: Test Fund\nKey fact: Lock-in period: 3 year(s)",
            fund_name="Test Fund",
            source_url="https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
            ingestion_timestamp="2026-08-30T06:21:03Z",
        ),
    ]


def test_embed_chunks_with_mock_model(sample_chunks: list[Chunk]):
    mock_embedder = MagicMock(spec=Embedder)
    mock_embedder.model_name = EMBEDDING_MODEL_NAME
    mock_embedder.dimension = EMBEDDING_DIMENSION
    mock_embedder.encode_texts.return_value = [
        np.random.rand(EMBEDDING_DIMENSION).tolist(),
        np.random.rand(EMBEDDING_DIMENSION).tolist(),
    ]

    embedded = embed_chunks(sample_chunks, embedder=mock_embedder)
    assert len(embedded) == 2
    assert embedded[0].chunk_id == "test-fund-001"
    assert embedded[0].embedding_model == EMBEDDING_MODEL_NAME
    assert embedded[0].embedding_dim == EMBEDDING_DIMENSION
    assert len(embedded[0].embedding) == EMBEDDING_DIMENSION
    mock_embedder.encode_texts.assert_called_once()


def test_embed_chunks_rejects_empty_list():
    with pytest.raises(EmbeddingError, match="empty chunk list"):
        embed_chunks([])


def test_embed_chunks_rejects_dimension_mismatch(sample_chunks: list[Chunk]):
    mock_embedder = MagicMock(spec=Embedder)
    mock_embedder.model_name = EMBEDDING_MODEL_NAME
    mock_embedder.dimension = 128
    mock_embedder.encode_texts.return_value = [
        [0.1] * 128,
        [0.2] * 128,
    ]

    with pytest.raises(EmbeddingError, match="Expected embedding dimension"):
        embed_chunks(sample_chunks, embedder=mock_embedder)


def test_encode_query_uses_same_model():
    embedder = Embedder()
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = EMBEDDING_DIMENSION
    mock_model.encode.return_value = np.array([[0.5] * EMBEDDING_DIMENSION])
    embedder._model = mock_model

    vector = embedder.encode_query("What is the expense ratio?")
    assert len(vector) == EMBEDDING_DIMENSION
    mock_model.encode.assert_called_once()


def test_load_chunks_from_file():
    chunks = load_chunks()
    assert len(chunks) > 0
    assert all(chunk.chunk_id for chunk in chunks)


def test_run_embedding_integration(monkeypatch, tmp_path, sample_chunks: list[Chunk]):
    mock_embedder = MagicMock(spec=Embedder)
    mock_embedder.model_name = EMBEDDING_MODEL_NAME
    mock_embedder.dimension = EMBEDDING_DIMENSION
    mock_embedder.encode_texts.return_value = [
        [0.1] * EMBEDDING_DIMENSION for _ in sample_chunks
    ]
    monkeypatch.setattr("src.ingestion.embedder.load_chunks", lambda path=None: sample_chunks)

    chunking_manifest_path = tmp_path / "metadata" / "chunking_manifest.json"
    chunking_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    chunking_manifest_path.write_text(
        json.dumps(
            {
                "status": "success",
                "chunking_timestamp": "2026-08-30T06:39:28Z",
                "total_chunks": len(sample_chunks),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.ingestion.embedder.CHUNKING_MANIFEST_PATH", chunking_manifest_path
    )
    monkeypatch.setattr("src.ingestion.embedder.EMBEDDINGS_DIR", tmp_path / "embeddings")
    monkeypatch.setattr("src.ingestion.embedder.METADATA_DIR", tmp_path / "metadata")
    monkeypatch.setattr(
        "src.ingestion.embedder.ALL_EMBEDDINGS_PATH",
        tmp_path / "embeddings" / "all_embeddings.json",
    )
    monkeypatch.setattr(
        "src.ingestion.embedder.EMBEDDING_MANIFEST_PATH",
        tmp_path / "metadata" / "embedding_manifest.json",
    )

    manifest = run_embedding(embedder=mock_embedder)
    assert manifest.status == "success"
    assert manifest.chunks_embedded == len(sample_chunks)
    assert manifest.embedding_model == EMBEDDING_MODEL_NAME
    assert manifest.embedding_dim == EMBEDDING_DIMENSION
