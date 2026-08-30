"""Phase 5 — Retrieval: embed question, top-k vector search, relevance check.

FR-7 (same embedding model at query time), FR-8 (top-k from ChromaDB),
FR-18 / E-3 (no relevant match → "not in my sources"), E-4 (ambiguous fund).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.config.corpus import CORPUS
from src.ingestion.embedder import Embedder
from src.ingestion.store import (
    RetrievedChunk,
    VectorStoreError,
    get_collection,
    get_collection_ingestion_timestamp,
)

TOP_K = 3
RELEVANCE_THRESHOLD = 0.4

# Alias fragments (lowercased) → canonical fund_name, for E-3/E-4 fund detection.
_FUND_ALIASES: list[tuple[str, str]] = []
for fund in CORPUS:
    _FUND_ALIASES.append((fund.fund_name.lower(), fund.fund_name))
    slug_alias = fund.slug.replace("-", " ")
    _FUND_ALIASES.append((slug_alias, fund.fund_name))
_FUND_ALIASES.extend(
    [
        ("hdfc elss", CORPUS[2].fund_name),
        ("elss", CORPUS[2].fund_name),
        ("tax saver", CORPUS[2].fund_name),
        ("flexi cap", CORPUS[1].fund_name),
        ("flexicap", CORPUS[1].fund_name),
        ("hdfc equity fund", CORPUS[1].fund_name),
        ("large cap fund", CORPUS[0].fund_name),
        ("largecap", CORPUS[0].fund_name),
        ("small cap fund", CORPUS[3].fund_name),
        ("smallcap", CORPUS[3].fund_name),
        ("balanced advantage", CORPUS[4].fund_name),
    ]
)

OUT_OF_CORPUS_PATTERN = re.compile(
    r"\bhdfc[\w ]*?(mid[ -]?cap|multi[ -]?cap|value|index|gilt|floating|"
    r"banking|infrastructure|infra|dividend|short[ -]?term|money[ -]?market|"
    r"liquid|arbitrage|contra|equity[ -]?savings|fo gold|opportunities)\b",
    re.IGNORECASE,
)

OUT_OF_SOURCES_MESSAGE = (
    "I don't have information about that fund in my sources. I can only "
    "answer questions about five HDFC funds: Large Cap, Flexi Cap, ELSS, "
    "Small Cap, and Balanced Advantage (Direct Growth)."
)

NO_MATCH_MESSAGE = (
    "I don't have that information in my sources. Try asking about expense "
    "ratio, lock-in period, minimum SIP, exit load, benchmark, or riskometer "
    "for one of the five HDFC funds."
)

AMBIGUOUS_MESSAGE = (
    "I have info on five HDFC funds: Large Cap, Flexi Cap, ELSS Tax Saver, "
    "Small Cap, and Balanced Advantage (Direct Growth). Which fund would you "
    "like me to look up?"
)


@dataclass
class RetrievalResult:
    question: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    cosine_scores: list[float] = field(default_factory=list)
    relevant: bool = False
    reason: str = "relevant"  # relevant | no_match | out_of_corpus | ambiguous
    message: str | None = None
    best_source_url: str | None = None
    last_updated: str | None = None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def contains_out_of_corpus_fund(question: str) -> bool:
    """Returns True if the question names an HDFC fund outside the corpus (E-3)."""
    return bool(OUT_OF_CORPUS_PATTERN.search(question))


def fund_mentions(question: str) -> list[str]:
    """Return canonical fund_names mentioned in the question (E-4)."""
    lower = question.lower()
    matches: list[str] = []
    for alias, fund_name in _FUND_ALIASES:
        if alias and alias in lower:
            matches.append(fund_name)
    return list(dict.fromkeys(matches))


def retrieve(
    question: str,
    top_k: int = TOP_K,
    threshold: float = RELEVANCE_THRESHOLD,
    embedder: Embedder | None = None,
) -> RetrievalResult:
    """Encode the question (FR-7) and search ChromaDB (FR-8), then gate on relevance.

    Pass a shared ``embedder`` (e.g. cached by the UI) to avoid re-loading the
    MiniLM model on every query.
    """
    if not question.strip():
        return RetrievalResult(question, reason="no_match", message=NO_MATCH_MESSAGE)

    if contains_out_of_corpus_fund(question):
        return RetrievalResult(
            question, reason="out_of_corpus", message=OUT_OF_SOURCES_MESSAGE
        )

    mentioned = fund_mentions(question)
    if not mentioned:
        return RetrievalResult(
            question, reason="ambiguous", message=AMBIGUOUS_MESSAGE
        )

    try:
        collection = get_collection()
        last_updated = get_collection_ingestion_timestamp()
    except VectorStoreError:
        last_updated = None
        raise

    encoder = embedder or Embedder()
    query_vector = encoder.encode_query(question)
    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_vector],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances", "embeddings"],
    }
    if len(mentioned) == 1:
        # FR-12: citation must match the fund asked about, so restrict search to
        # that fund's chunks when the question names exactly one fund.
        query_kwargs["where"] = {"fund_name": mentioned[0]}
    results = collection.query(**query_kwargs)

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    embeddings = results.get("embeddings", [[]])[0]

    chunks: list[RetrievedChunk] = []
    scores: list[float] = []
    for chunk_id, text, metadata, distance, vector in zip(
        ids, documents, metadatas, distances, embeddings, strict=True
    ):
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=text or "",
                fund_name=metadata.get("fund_name", ""),
                source_url=metadata.get("source_url", ""),
                ingestion_timestamp=metadata.get("ingestion_timestamp", ""),
                distance=float(distance),
            )
        )
        scores.append(_cosine(query_vector, list(vector)))

    if not chunks:
        return RetrievalResult(
            question, reason="no_match", message=NO_MATCH_MESSAGE
        )

    best_index = max(range(len(scores)), key=scores.__getitem__)
    relevant = scores[best_index] >= threshold
    if not relevant:
        return RetrievalResult(
            question, chunks=chunks, cosine_scores=scores,
            relevant=False, reason="no_match", message=NO_MATCH_MESSAGE,
        )

    result = RetrievalResult(
        question=question,
        chunks=chunks,
        cosine_scores=scores,
        relevant=True,
        reason="relevant",
        best_source_url=chunks[best_index].source_url,
        last_updated=last_updated,
    )
    return result


def format_retrieval(result: RetrievalResult) -> str:
    """Human-readable dump of a RetrievalResult for CLI/testing."""
    if result.message:
        return f"[{result.reason}] {result.message}"
    lines = [
        f"Question: {result.question}",
        f"Reason: {result.reason} | relevant={result.relevant}",
    ]
    if result.last_updated:
        lines.append(f"Last updated from sources: {result.last_updated}")
    for chunk, score in zip(result.chunks, result.cosine_scores, strict=True):
        lines.append(
            f"  score={score:.3f} dist={chunk.distance:.3f} "
            f"[{chunk.chunk_id}] fund={chunk.fund_name}\n    {chunk.text[:160]!r}"
        )
    return "\n".join(lines)