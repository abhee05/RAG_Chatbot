"""Tests for Phase 2 chunking."""

import json

import pytest

from src.config.corpus import ALLOWED_SOURCE_URLS, CORPUS
from src.ingestion.chunker import (
    ChunkingError,
    _parse_fact_lines,
    _parse_faq_pairs,
    _split_with_overlap,
    chunk_document,
    load_document,
    run_chunking,
)
from src.ingestion.loader import LoadedDocument


@pytest.fixture
def sample_document() -> LoadedDocument:
    return LoadedDocument(
        fund_name="HDFC ELSS Tax Saver Fund Direct Plan Growth",
        category="ELSS",
        source_url=CORPUS[2].source_url,
        raw_text=(
            "Fund: HDFC ELSS Tax Saver Fund Direct Plan Growth\n"
            "Category: ELSS\n"
            "Source URL: https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth\n\n"
            "Key fund facts:\n"
            "- Expense ratio: 1.19%\n"
            "- Lock-in period: 3 year(s)\n"
            "- Minimum SIP investment: 500\n\n"
            "FAQ:\n"
            "Q: How much expense ratio is charged?\n"
            "A: The Expense Ratio is 1.19%.\n\n"
            "Q: What is the lock-in?\n"
            "A: ELSS has a 3 year lock-in.\n\n"
            "Additional page content:\n"
            "Minimum SIP Investment is set to 500."
        ),
        ingestion_timestamp="2026-08-30T06:21:03Z",
        slug="hdfc-elss-tax-saver-fund-direct-plan-growth",
    )


def test_parse_fact_lines():
    facts = _parse_fact_lines("- Expense ratio: 1.19%\n- Exit load: Nil")
    assert facts == ["Expense ratio: 1.19%", "Exit load: Nil"]


def test_parse_faq_pairs():
    text = "Q: What is the lock-in?\nA: 3 years.\n\nQ: Min SIP?\nA: 500."
    pairs = _parse_faq_pairs(text)
    assert len(pairs) == 2
    assert pairs[0][0] == "What is the lock-in?"


def test_split_with_overlap():
    text = "word " * 200
    chunks = _split_with_overlap(text, max_chars=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_chunk_document_semantic_sections(sample_document: LoadedDocument):
    chunks = chunk_document(sample_document)
    assert len(chunks) >= 5

    chunk_ids = [chunk.chunk_id for chunk in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))
    assert all(chunk.chunk_id.startswith(sample_document.slug) for chunk in chunks)
    assert all(chunk.source_url == sample_document.source_url for chunk in chunks)
    assert all(chunk.fund_name == sample_document.fund_name for chunk in chunks)
    assert all(chunk.source_url in ALLOWED_SOURCE_URLS for chunk in chunks)

    combined = "\n".join(chunk.text for chunk in chunks)
    assert "Expense ratio: 1.19%" in combined
    assert "Lock-in period: 3 year(s)" in combined
    assert "FAQ Question: How much expense ratio is charged?" in combined


def test_chunk_document_rejects_unknown_source():
    document = LoadedDocument(
        fund_name="Unknown",
        category="Test",
        source_url="https://groww.in/mutual-funds/unknown",
        raw_text="Key fund facts:\n- Expense ratio: 1%",
        ingestion_timestamp="2026-08-30T06:21:03Z",
        slug="unknown",
    )
    with pytest.raises(ChunkingError, match="not in corpus"):
        chunk_document(document)


def test_load_document_from_file(tmp_path):
    payload = {
        "fund_name": "HDFC Large Cap Fund Direct Growth",
        "category": "Large-cap",
        "source_url": CORPUS[0].source_url,
        "raw_text": "Key fund facts:\n- Expense ratio: 1.0%",
        "ingestion_timestamp": "2026-08-30T06:21:03Z",
        "slug": "hdfc-large-cap-fund-direct-growth",
    }
    path = tmp_path / "doc.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    document = load_document(path)
    assert document.slug == "hdfc-large-cap-fund-direct-growth"


def test_run_chunking_integration(monkeypatch, tmp_path):
    monkeypatch.setattr("src.ingestion.chunker.CHUNKS_DIR", tmp_path / "chunks")
    monkeypatch.setattr(
        "src.ingestion.chunker.ALL_CHUNKS_PATH",
        tmp_path / "chunks" / "all_chunks.json",
    )
    monkeypatch.setattr("src.ingestion.chunker.METADATA_DIR", tmp_path / "metadata")
    monkeypatch.setattr(
        "src.ingestion.chunker.CHUNKING_MANIFEST_PATH",
        tmp_path / "metadata" / "chunking_manifest.json",
    )

    manifest = run_chunking()
    assert manifest.status == "success"
    assert manifest.documents_chunked == 5
    assert manifest.total_chunks > 0
    assert all(fund.chunk_count > 0 for fund in manifest.funds if fund.status == "success")
    assert (tmp_path / "chunks" / "all_chunks.json").exists()
    assert (tmp_path / "metadata" / "chunking_manifest.json").exists()
