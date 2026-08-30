"""Tests for Phase 1 data loading."""

from pathlib import Path

import pytest

from src.config.corpus import CORPUS
from src.ingestion.loader import (
    DataLoadError,
    _build_raw_text,
    _extract_faq_entries,
    _extract_next_data,
    parse_groww_page,
    validate_source_url,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_html() -> str:
    path = FIXTURES_DIR / "sample_groww_page.html"
    if path.exists():
        return path.read_text(encoding="utf-8")
    pytest.skip("sample HTML fixture not available")


def test_corpus_has_five_funds():
    assert len(CORPUS) == 5


def test_validate_source_url_rejects_unknown():
    with pytest.raises(DataLoadError, match="not in allowed corpus"):
        validate_source_url("https://groww.in/mutual-funds/other-fund")


def test_validate_source_url_accepts_corpus_url():
    validate_source_url(CORPUS[0].source_url)


def test_extract_next_data(sample_html: str):
    data = _extract_next_data(sample_html)
    assert "props" in data


def test_parse_groww_page(sample_html: str):
    mf_data, raw_text = parse_groww_page(sample_html, CORPUS[2])
    assert mf_data["scheme_name"]
    assert "Expense ratio" in raw_text
    assert "Lock-in period" in raw_text
    assert "Minimum SIP investment" in raw_text
    assert "Benchmark" in raw_text
    assert "Riskometer" in raw_text


def test_extract_faq_entries(sample_html: str):
    faqs = _extract_faq_entries(sample_html)
    assert len(faqs) >= 1
    assert all(question and answer for question, answer in faqs)


def test_build_raw_text_includes_factual_fields():
    mf_data = {
        "scheme_name": "HDFC ELSS Tax Saver Fund Direct Plan Growth",
        "expense_ratio": "1.19",
        "exit_load": "Nil",
        "lock_in": {"years": 3, "months": 0, "days": 0},
        "min_sip_investment": 500,
        "benchmark": "NIFTY 500 TRI",
        "nfo_risk": "Moderately High",
    }
    raw_text = _build_raw_text(CORPUS[2], mf_data, [], "")
    assert "Expense ratio: 1.19%" in raw_text
    assert "Lock-in period: 3 year(s)" in raw_text
    assert "Minimum SIP investment: 500" in raw_text
    assert "Riskometer: Moderately High" in raw_text
