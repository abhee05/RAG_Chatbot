"""Phase 2 — Chunking: split loaded documents into retrieval-friendly segments."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config.corpus import ALLOWED_SOURCE_URLS
from src.config.paths import (
    ALL_CHUNKS_PATH,
    CHUNKING_MANIFEST_PATH,
    CHUNKS_DIR,
    INGESTION_MANIFEST_PATH,
    METADATA_DIR,
    RAW_DOCUMENTS_DIR,
)
from src.ingestion.loader import LoadedDocument

SECTION_KEY_FACTS = "Key fund facts:"
SECTION_FAQ = "FAQ:"
SECTION_ADDITIONAL = "Additional page content:"

MAX_CHUNK_CHARS = 500
OVERLAP_CHARS = 80

FAQ_BLOCK_PATTERN = re.compile(
    r"Q:\s*(?P<question>.+?)\s*A:\s*(?P<answer>.+?)(?=\n\nQ:|\Z)",
    re.DOTALL,
)


class ChunkingError(Exception):
    """Raised when documents cannot be chunked."""


@dataclass
class Chunk:
    chunk_id: str
    text: str
    fund_name: str
    source_url: str
    ingestion_timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FundChunkResult:
    slug: str
    source_url: str
    status: str
    chunk_count: int = 0
    error: str | None = None
    output_path: str | None = None


@dataclass
class ChunkingManifest:
    chunking_timestamp: str
    ingestion_timestamp: str | None
    status: str
    documents_chunked: int
    documents_failed: int
    total_chunks: int
    funds: list[FundChunkResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunking_timestamp": self.chunking_timestamp,
            "ingestion_timestamp": self.ingestion_timestamp,
            "status": self.status,
            "documents_chunked": self.documents_chunked,
            "documents_failed": self.documents_failed,
            "total_chunks": self.total_chunks,
            "funds": [asdict(fund) for fund in self.funds],
        }


def ensure_output_dirs() -> None:
    for path in (CHUNKS_DIR, METADATA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_document(path: Path) -> LoadedDocument:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LoadedDocument(
        fund_name=payload["fund_name"],
        category=payload.get("category", ""),
        source_url=payload["source_url"],
        raw_text=payload["raw_text"],
        ingestion_timestamp=payload["ingestion_timestamp"],
        slug=payload["slug"],
    )


def _validate_document(document: LoadedDocument) -> None:
    if document.source_url not in ALLOWED_SOURCE_URLS:
        raise ChunkingError(
            f"Document source URL not in corpus: {document.source_url}"
        )
    if not document.raw_text.strip():
        raise ChunkingError(f"Document {document.slug} has empty raw_text")


def _split_with_overlap(text: str, max_chars: int, overlap: int) -> list[str]:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return [normalized]

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        chunks.append(normalized[start:end].strip())
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


def _extract_section(raw_text: str, header: str, next_headers: tuple[str, ...]) -> str:
    start = raw_text.find(header)
    if start == -1:
        return ""

    content_start = start + len(header)
    end = len(raw_text)
    for next_header in next_headers:
        idx = raw_text.find(next_header, content_start)
        if idx != -1:
            end = min(end, idx)
    return raw_text[content_start:end].strip()


def _parse_fact_lines(section_text: str) -> list[str]:
    facts: list[str] = []
    for line in section_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            facts.append(stripped[2:].strip())
    return facts


def _parse_faq_pairs(section_text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for match in FAQ_BLOCK_PATTERN.finditer(section_text):
        question = " ".join(match.group("question").split())
        answer = " ".join(match.group("answer").split())
        if question and answer:
            pairs.append((question, answer))
    return pairs


def _chunk_texts_for_section(section_texts: list[str]) -> list[str]:
    chunks: list[str] = []
    for section_text in section_texts:
        if len(section_text) <= MAX_CHUNK_CHARS:
            chunks.append(section_text)
        else:
            chunks.extend(
                _split_with_overlap(section_text, MAX_CHUNK_CHARS, OVERLAP_CHARS)
            )
    return chunks


def _with_fund_context(document: LoadedDocument, body: str) -> str:
    return (
        f"Fund: {document.fund_name}\n"
        f"Source URL: {document.source_url}\n"
        f"{body}"
    )


def chunk_document(document: LoadedDocument) -> list[Chunk]:
    _validate_document(document)

    raw_text = document.raw_text
    section_texts: list[str] = []

    key_facts = _extract_section(
        raw_text,
        SECTION_KEY_FACTS,
        (SECTION_FAQ, SECTION_ADDITIONAL),
    )
    for fact in _parse_fact_lines(key_facts):
        section_texts.append(f"Key fact: {fact}")

    faq_section = _extract_section(raw_text, SECTION_FAQ, (SECTION_ADDITIONAL,))
    for question, answer in _parse_faq_pairs(faq_section):
        section_texts.append(f"FAQ Question: {question}\nFAQ Answer: {answer}")

    additional = _extract_section(raw_text, SECTION_ADDITIONAL, ())
    if additional:
        section_texts.extend(_chunk_texts_for_section([additional]))

    if not section_texts:
        section_texts.extend(_chunk_texts_for_section([raw_text]))

    chunks: list[Chunk] = []
    for index, section_text in enumerate(section_texts, start=1):
        chunk_id = f"{document.slug}-{index:03d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=_with_fund_context(document, section_text),
                fund_name=document.fund_name,
                source_url=document.source_url,
                ingestion_timestamp=document.ingestion_timestamp,
            )
        )

    return chunks


def save_fund_chunks(slug: str, chunks: list[Chunk]) -> Path:
    path = CHUNKS_DIR / f"{slug}.json"
    payload = [chunk.to_dict() for chunk in chunks]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return path


def save_all_chunks(all_chunks: list[Chunk]) -> Path:
    payload = [chunk.to_dict() for chunk in all_chunks]
    ALL_CHUNKS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return ALL_CHUNKS_PATH


def save_manifest(manifest: ChunkingManifest) -> Path:
    CHUNKING_MANIFEST_PATH.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return CHUNKING_MANIFEST_PATH


def _read_ingestion_timestamp() -> str | None:
    if not INGESTION_MANIFEST_PATH.exists():
        return None
    payload = json.loads(INGESTION_MANIFEST_PATH.read_text(encoding="utf-8"))
    return payload.get("ingestion_timestamp")


def _assert_ingestion_available() -> None:
    if not INGESTION_MANIFEST_PATH.exists():
        raise ChunkingError(
            "Ingestion manifest not found. Run Phase 1 data loading first."
        )

    manifest = json.loads(INGESTION_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") == "failed":
        raise ChunkingError(
            "Ingestion failed. Downstream chunking is blocked (E-8)."
        )

    if not manifest.get("documents_loaded"):
        raise ChunkingError("No documents available to chunk.")


def run_chunking() -> ChunkingManifest:
    from src.ingestion.loader import utc_now_iso

    ensure_output_dirs()
    _assert_ingestion_available()

    document_paths = sorted(RAW_DOCUMENTS_DIR.glob("*.json"))
    if not document_paths:
        raise ChunkingError(
            f"No documents found in {RAW_DOCUMENTS_DIR}. Run Phase 1 first."
        )

    chunking_timestamp = utc_now_iso()
    ingestion_timestamp = _read_ingestion_timestamp()
    results: list[FundChunkResult] = []
    all_chunks: list[Chunk] = []
    chunked_count = 0

    for path in document_paths:
        try:
            document = load_document(path)
            chunks = chunk_document(document)
            output_path = save_fund_chunks(document.slug, chunks)
            all_chunks.extend(chunks)
            chunked_count += 1
            results.append(
                FundChunkResult(
                    slug=document.slug,
                    source_url=document.source_url,
                    status="success",
                    chunk_count=len(chunks),
                    output_path=str(output_path),
                )
            )
        except Exception as exc:  # noqa: BLE001 - collect per-fund failures
            slug = path.stem
            source_url = ""
            try:
                source_url = load_document(path).source_url
            except Exception:
                pass
            results.append(
                FundChunkResult(
                    slug=slug,
                    source_url=source_url,
                    status="failed",
                    error=str(exc),
                )
            )

    failed_count = len(document_paths) - chunked_count
    if chunked_count == len(document_paths):
        status = "success"
    elif chunked_count == 0:
        status = "failed"
    else:
        status = "partial"

    if all_chunks:
        save_all_chunks(all_chunks)

    manifest = ChunkingManifest(
        chunking_timestamp=chunking_timestamp,
        ingestion_timestamp=ingestion_timestamp,
        status=status,
        documents_chunked=chunked_count,
        documents_failed=failed_count,
        total_chunks=len(all_chunks),
        funds=results,
    )
    save_manifest(manifest)

    if status == "failed":
        raise ChunkingError("All documents failed to chunk.")

    return manifest


def main() -> None:
    manifest = run_chunking()
    print(
        f"Chunking {manifest.status}: "
        f"{manifest.documents_chunked} documents, "
        f"{manifest.total_chunks} total chunks."
    )
    print(f"Manifest: {CHUNKING_MANIFEST_PATH}")
    print(f"Combined chunks: {ALL_CHUNKS_PATH}")
    if manifest.documents_failed:
        for fund in manifest.funds:
            if fund.status == "failed":
                print(f"  FAILED {fund.slug}: {fund.error}")


if __name__ == "__main__":
    main()
