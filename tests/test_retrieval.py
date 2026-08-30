"""Tests for Phase 5 retrieval backend (gates, retriever, prompt, answer)."""

import pytest

from src.ingestion.embedder import Embedder
from src.ingestion.store import RetrievedChunk
from src.retrieval.answer import AnswerResult, answer_question, format_answer
from src.retrieval.gates import (
    evaluate_gates,
    is_advice_request,
    is_empty_or_gibberish,
    is_returns_comparison,
)
from src.retrieval.prompt import build_messages
from src.retrieval.retriever import (
    contains_out_of_corpus_fund,
    fund_mentions,
    retrieve,
)


def sample_chunk(text: str = "Key fact: Expense ratio: 1.19%") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="test-001",
        text=text,
        fund_name="HDFC ELSS Tax Saver Fund Direct Plan Growth",
        source_url="https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
        ingestion_timestamp="2026-08-30T06:21:03Z",
        distance=0.5,
    )


# ---------------------------------------------------------------- gates (E-1..E-6)

def test_gate_allows_factual_question():
    result = evaluate_gates("What is the expense ratio of HDFC ELSS?")
    assert result.allowed is True
    assert result.reason is None


def test_gate_blocks_advice():
    result = evaluate_gates("Should I buy HDFC Small Cap Fund?")
    assert result.allowed is False
    assert result.reason == "advice"
    assert "not investment advice" in (result.message or "")
    assert is_advice_request("Is it good to invest now?") is True


def test_gate_blocks_returns_comparison():
    for question in (
        "Which fund gave better returns, Large Cap or Flexi Cap?",
        "Compare returns of HDFC ELSS vs HDFC Small Cap",
    ):
        assert evaluate_gates(question).reason == "returns"
    assert is_returns_comparison("Which fund performed best last year?") is True


def test_gate_allows_single_fund_returns_fact():
    # "what returns does X provide" is answered factually from the corpus FAQ.
    assert evaluate_gates("What returns does HDFC ELSS provide?").allowed is True


def test_gate_blocks_pii():
    for question in (
        "My PAN is ABCDE1234F, can you help?",
        "My Aadhaar is 1234 5678 9012",
        "Call me at 9876504321",
        "Email me at foo@bar.com",
    ):
        result = evaluate_gates(question)
        assert result.allowed is False
        assert result.reason == "pii"
        assert "personal information" in (result.message or "")


def test_gate_reprompts_empty_and_gibberish():
    assert evaluate_gates("").reason == "invalid"
    assert evaluate_gates("   ").reason == "invalid"
    assert evaluate_gates("asdfgh").reason == "invalid"
    assert is_empty_or_gibberish("12345") is True


def test_advice_pii_priority():
    # PII outranks advice.
    result = evaluate_gates("Should I buy with PAN ABCDE1234F?")
    assert result.reason == "pii"


# ---------------------------------------------------------------- retriever (E-3, E-4)

def test_retriever_out_of_corpus_fund():
    assert contains_out_of_corpus_fund("HDFC Mid Cap Fund") is True
    assert contains_out_of_corpus_fund("HDFC ELSS Tax Saver Fund") is False
    result = retrieve("What is the expense ratio of HDFC Mid Cap Fund?")
    assert result.reason == "out_of_corpus"
    assert "in my sources" in (result.message or "")


def test_retriever_ambiguous_fund_asks_to_clarify():
    result = retrieve("What is the expense ratio?")
    assert result.reason == "ambiguous"
    assert "Which fund" in (result.message or "")


def test_fund_mentions_maps_aliases():
    mentions = fund_mentions("What is the lock-in period for ELSS?")
    assert mentions == ["HDFC ELSS Tax Saver Fund Direct Plan Growth"]


def test_retriever_factual_query_relevant():
    result = retrieve("What is the expense ratio of HDFC ELSS Tax Saver Fund?")
    assert result.reason == "relevant"
    assert result.relevant is True
    assert result.best_source_url == (
        "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth"
    )
    assert result.chunks
    assert all(
        c.fund_name == "HDFC ELSS Tax Saver Fund Direct Plan Growth"
        for c in result.chunks
    )
    assert result.last_updated is not None


def test_retriever_restricts_to_named_fund():
    result = retrieve("What is the NAV of HDFC Small Cap Fund?", top_k=2)
    assert all(c.fund_name == "HDFC Small Cap Fund Direct Growth" for c in result.chunks)


# ---------------------------------------------------------------- prompt (FR-9)

def test_build_messages_includes_question_and_context():
    chunk = sample_chunk()
    messages = build_messages("What is the expense ratio?", [chunk])
    assert messages[0]["role"] == "system"
    assert "at most 3 sentences" in messages[0]["content"]
    user = messages[1]["content"]
    assert "What is the expense ratio?" in user
    assert "Expense ratio" in user
    assert "groww.in/mutual-funds" in user


def test_build_messages_rejects_empty_context():
    with pytest.raises(ValueError, match="zero retrieved chunks"):
        build_messages("Question?", [])


# ---------------------------------------------------------------- answer (FR-12, FR-13)

def test_answer_blocks_before_llm(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("LLM should not be called for blocked queries")

    monkeypatch.setattr("src.retrieval.answer.generate_answer", _boom)
    result = answer_question("Should I invest in HDFC Small Cap?", call_llm=True)
    assert result.status == "blocked"
    assert result.reason == "advice"


def test_answer_retrieval_only_mode():
    result = answer_question(
        "What is the exit load on HDFC Small Cap Fund?", call_llm=False
    )
    assert result.status == "answered"
    assert result.source_url == (
        "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth"
    )
    assert result.last_updated is not None
    assert "hdfc-small-cap-fund-direct-growth" in result.message


def test_answer_out_of_corpus():
    result = answer_question("HDFC Mid Cap expense ratio?", call_llm=False)
    assert result.status == "out_of_corpus"


def test_format_answer_appends_citation_and_date():
    text = format_answer(
        "The expense ratio is 1.19%.",
        "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
        "2026-08-30T06:21:03Z",
    )
    assert "The expense ratio is 1.19%." in text
    assert "Source: https://groww.in/mutual-funds/" in text
    assert "Last updated from sources: August 30, 2026" in text


def test_answer_result_serializable():
    result = answer_question("What is the expense ratio of HDFC ELSS?", call_llm=False)
    assert isinstance(result, AnswerResult)
    payload = result.to_dict()
    assert payload["status"] == "answered"
    assert "retrieval" in payload