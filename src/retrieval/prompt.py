"""Phase 5 — RAG prompt assembly (FR-9, architecture.md §7.3).

System rules enforce: facts-only, no investment advice, answer strictly from
retrieved chunks, ≤3 sentences, no returns computation, no citations (the
orchestrator appends a single citation + last-updated line).
"""

from __future__ import annotations

from src.ingestion.store import RetrievedChunk

SYSTEM_PROMPT = (
    "You are a facts-only mutual fund FAQ assistant for five HDFC funds "
    "(Large Cap, Flexi Cap, ELSS Tax Saver, Small Cap, Balanced Advantage — "
    "Direct Growth).\n"
    "Rules:\n"
    "- Answer ONLY from the provided context chunks.\n"
    "- State facts exactly as given; never invent numbers or details.\n"
    "- Give no investment advice, opinions, or recommendations.\n"
    "- Do not compute or compare returns or performance.\n"
    "- Keep the answer to at most 3 sentences.\n"
    "- Do not add citations, source URLs, or 'Last updated' lines; those are "
    "added separately."
)


def context_from_chunks(chunks: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{index}] Fund: {chunk.fund_name}\n"
            f"Source URL: {chunk.source_url}\n"
            f"Content: {chunk.text}"
        )
    return "\n\n".join(blocks)


def build_messages(
    question: str,
    chunks: list[RetrievedChunk],
) -> list[dict[str, str]]:
    if not chunks:
        raise ValueError("Cannot build a RAG prompt with zero retrieved chunks.")
    user_content = (
        f"Question: {question}\n\n"
        "Context (answer only from this):\n"
        f"{context_from_chunks(chunks)}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]