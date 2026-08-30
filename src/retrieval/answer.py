"""Phase 5 — Query orchestration: gates → retrieve → prompt → LLM → post-process.

Implements the pipeline in architecture.md §7.1 and post-processing rules
FR-11 (≤3 sentences enforced in prompt), FR-12 (one citation), FR-13
("Last updated from sources: <date>").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from src.ingestion.store import RetrievedChunk
from src.retrieval.gates import evaluate_gates
from src.retrieval.llm import LLMError, generate_answer
from src.retrieval.prompt import build_messages
from src.retrieval.retriever import TOP_K, RetrievalResult, retrieve

LLM_RETRY_MESSAGE = (
    "Sorry, the answer service is unavailable right now (E-9). "
    "Please try again in a moment — I won't guess without my sources."
)


@dataclass
class AnswerResult:
    status: str  # answered | blocked | no_match | out_of_corpus | ambiguous | error
    message: str
    source_url: str | None = None
    last_updated: str | None = None
    reason: str | None = None
    retrieval: RetrievalResult | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.retrieval is not None:
            payload["retrieval"] = {
                "reason": self.retrieval.reason,
                "relevant": self.retrieval.relevant,
                "cosine_scores": self.retrieval.cosine_scores,
                "chunk_ids": [c.chunk_id for c in self.retrieval.chunks],
                "fund_names": [c.fund_name for c in self.retrieval.chunks],
            }
        return payload


def _parse_date(timestamp: str | None) -> str | None:
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc).strftime("%B %d, %Y")


def _dedupe_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Drop duplicate chunk IDs so the prompt stays tight."""
    seen: set[str] = set()
    filtered: list[RetrievedChunk] = []
    for chunk in chunks:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            filtered.append(chunk)
    return filtered


def format_answer(answer: str, source_url: str | None, last_updated: str | None) -> str:
    lines = [answer.strip()]
    if source_url:
        lines.append(f"\nSource: {source_url}")
    formatted_date = _parse_date(last_updated)
    if formatted_date:
        lines.append(f"Last updated from sources: {formatted_date}")
    return "\n".join(lines)


def answer_question(
    question: str,
    top_k: int = TOP_K,
    call_llm: bool = True,
    embedder: Any | None = None,
) -> AnswerResult:
    """Full Phase 5 pipeline for a single user question."""
    gate = evaluate_gates(question)
    if not gate.allowed:
        return AnswerResult(
            status="blocked",
            message=gate.message or "",
            reason=gate.reason,
        )

    result = retrieve(question, top_k=top_k, embedder=embedder)
    if result.reason != "relevant":
        return AnswerResult(
            status=result.reason,
            message=result.message or "",
            reason=result.reason,
            retrieval=result,
        )

    chunks = _dedupe_chunks(result.chunks)
    messages = build_messages(question, chunks)

    if not call_llm:
        preview = "\n".join(f" - {chunk.chunk_id} ({chunk.fund_name})" for chunk in chunks)
        message = (
            "Retrieval-only mode (no LLM call). Top chunks:\n"
            f"{preview}\n\n"
            "Answer + citation + last-updated would be generated via Mistral here."
        )
        return AnswerResult(
            status="answered",
            message=message,
            source_url=result.best_source_url,
            last_updated=result.last_updated,
            retrieval=result,
        )

    try:
        answer = generate_answer(messages)
    except LLMError:
        return AnswerResult(
            status="error",
            message=LLM_RETRY_MESSAGE,
            reason="llm_error",
            retrieval=result,
        )

    message = format_answer(answer, result.best_source_url, result.last_updated)
    return AnswerResult(
        status="answered",
        message=message,
        source_url=result.best_source_url,
        last_updated=result.last_updated,
        reason="answered",
        retrieval=result,
    )