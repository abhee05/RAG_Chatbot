"""Phase 1 — Data Loading: fetch Groww fund pages and persist raw artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

from src.config.corpus import ALLOWED_SOURCE_URLS, CORPUS, FundSource
from src.config.paths import (
    INGESTION_MANIFEST_PATH,
    METADATA_DIR,
    RAW_DOCUMENTS_DIR,
    RAW_HTML_DIR,
)

USER_AGENT = (
    "Mozilla/5.0 (compatible; HDFCFundFAQBot/1.0; +https://groww.in/)"
)
REQUEST_TIMEOUT_SECONDS = 30

FACTUAL_FIELD_LABELS: tuple[tuple[str, str], ...] = (
    ("scheme_name", "Scheme name"),
    ("fund_name", "Fund name"),
    ("category", "Category"),
    ("sub_category", "Sub-category"),
    ("plan_type", "Plan type"),
    ("expense_ratio", "Expense ratio"),
    ("exit_load", "Exit load"),
    ("lock_in", "Lock-in period"),
    ("min_sip_investment", "Minimum SIP investment"),
    ("min_investment_amount", "Minimum investment amount"),
    ("benchmark", "Benchmark"),
    ("benchmark_name", "Benchmark name"),
    ("nfo_risk", "Riskometer"),
    ("description", "Description"),
    ("launch_date", "Launch date"),
    ("fund_manager", "Fund manager"),
    ("fund_house", "Fund house"),
    ("stamp_duty", "Stamp duty"),
    ("sip_allowed", "SIP allowed"),
    ("lumpsum_allowed", "Lumpsum allowed"),
)


class DataLoadError(Exception):
    """Raised when a corpus URL cannot be loaded or parsed."""


@dataclass
class LoadedDocument:
    fund_name: str
    category: str
    source_url: str
    raw_text: str
    ingestion_timestamp: str
    slug: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FundLoadResult:
    slug: str
    source_url: str
    status: str
    error: str | None = None
    document_path: str | None = None
    html_path: str | None = None


@dataclass
class IngestionManifest:
    ingestion_timestamp: str
    status: str
    documents_loaded: int
    documents_failed: int
    funds: list[FundLoadResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ingestion_timestamp": self.ingestion_timestamp,
            "status": self.status,
            "documents_loaded": self.documents_loaded,
            "documents_failed": self.documents_failed,
            "funds": [asdict(fund) for fund in self.funds],
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def ensure_output_dirs() -> None:
    for path in (RAW_HTML_DIR, RAW_DOCUMENTS_DIR, METADATA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def validate_source_url(source_url: str) -> None:
    if source_url not in ALLOWED_SOURCE_URLS:
        raise DataLoadError(
            f"URL not in allowed corpus: {source_url}. "
            "Only the five Groww HDFC fund pages are permitted."
        )


def fetch_html(source_url: str, session: requests.Session | None = None) -> str:
    validate_source_url(source_url)
    client = session or requests.Session()
    response = client.get(
        source_url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    if "groww.in" not in response.url:
        raise DataLoadError(f"Unexpected redirect away from Groww: {response.url}")
    return response.text


def _extract_next_data(html: str) -> dict[str, Any]:
    marker = html.find("__NEXT_DATA__")
    if marker == -1:
        raise DataLoadError("Could not find __NEXT_DATA__ payload in HTML")

    start = html.find("{", marker)
    end = html.find("</script>", marker)
    if start == -1 or end == -1:
        raise DataLoadError("Malformed __NEXT_DATA__ script block")

    return json.loads(html[start:end])


def _strip_html(text: str) -> str:
    cleaned = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", unescape(cleaned)).strip()


def _format_lock_in(lock_in: Any) -> str:
    if not isinstance(lock_in, dict):
        return str(lock_in)
    parts: list[str] = []
    if lock_in.get("years"):
        parts.append(f"{lock_in['years']} year(s)")
    if lock_in.get("months"):
        parts.append(f"{lock_in['months']} month(s)")
    if lock_in.get("days"):
        parts.append(f"{lock_in['days']} day(s)")
    return ", ".join(parts) if parts else "None"


def _format_value(key: str, value: Any) -> str:
    if value is None:
        return "Not available"
    if key == "lock_in":
        return _format_lock_in(value)
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if key == "expense_ratio":
        text = str(value).strip()
        return f"{text}%" if text and not text.endswith("%") else text
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return str(value).strip()


def _extract_faq_entries(html: str) -> list[tuple[str, str]]:
    faqs: list[tuple[str, str]] = []
    soup = BeautifulSoup(html, "html.parser")
    for script in soup.find_all("script", type="application/ld+json"):
        if not script.string:
            continue
        try:
            payload = json.loads(script.string)
        except json.JSONDecodeError:
            continue
        if payload.get("@type") != "FAQPage":
            continue
        for entity in payload.get("mainEntity", []):
            question = entity.get("name")
            answer = entity.get("acceptedAnswer", {}).get("text")
            if question and answer:
                faqs.append((_strip_html(question), _strip_html(answer)))
    return faqs


def _build_raw_text(
    fund: FundSource,
    mf_data: dict[str, Any],
    faqs: list[tuple[str, str]],
    visible_text: str,
) -> str:
    sections: list[str] = [
        f"Fund: {fund.fund_name}",
        f"Category: {fund.category}",
        f"Source URL: {fund.source_url}",
        "",
        "Key fund facts:",
    ]

    for field_key, label in FACTUAL_FIELD_LABELS:
        if field_key not in mf_data:
            continue
        sections.append(f"- {label}: {_format_value(field_key, mf_data[field_key])}")

    if faqs:
        sections.extend(["", "FAQ:"])
        for question, answer in faqs:
            sections.extend([f"Q: {question}", f"A: {answer}", ""])

    if visible_text:
        sections.extend(["", "Additional page content:", visible_text])

    return "\n".join(sections).strip()


def _extract_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    chunks: list[str] = []
    for node in soup.find_all(string=re.compile(r"(?i)expense ratio|riskometer|exit load|benchmark|minimum sip|lock-in")):
        text = " ".join(node.strip().split())
        if text and len(text) > 20:
            chunks.append(text)

    deduped: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk not in seen:
            seen.add(chunk)
            deduped.append(chunk)

    return "\n".join(deduped[:20])


def parse_groww_page(html: str, fund: FundSource) -> tuple[dict[str, Any], str]:
    next_data = _extract_next_data(html)
    mf_data = next_data.get("props", {}).get("pageProps", {}).get("mfServerSideData")
    if not mf_data:
        raise DataLoadError("mfServerSideData missing from Groww page payload")

    faqs = _extract_faq_entries(html)
    visible_text = _extract_visible_text(html)
    raw_text = _build_raw_text(fund, mf_data, faqs, visible_text)
    return mf_data, raw_text


def save_html_snapshot(html: str, slug: str) -> Path:
    path = RAW_HTML_DIR / f"{slug}.html"
    path.write_text(html, encoding="utf-8")
    return path


def save_document(document: LoadedDocument) -> Path:
    path = RAW_DOCUMENTS_DIR / f"{document.slug}.json"
    path.write_text(
        json.dumps(document.to_dict(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return path


def save_manifest(manifest: IngestionManifest) -> Path:
    path = INGESTION_MANIFEST_PATH
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_fund(
    fund: FundSource,
    ingestion_timestamp: str,
    session: requests.Session | None = None,
) -> LoadedDocument:
    html = fetch_html(fund.source_url, session=session)
    _, raw_text = parse_groww_page(html, fund)

    slug = fund.slug
    save_html_snapshot(html, slug)

    scheme_name = fund.fund_name
    document = LoadedDocument(
        fund_name=scheme_name,
        category=fund.category,
        source_url=fund.source_url,
        raw_text=raw_text,
        ingestion_timestamp=ingestion_timestamp,
        slug=slug,
    )
    save_document(document)
    return document


def run_ingestion(session: requests.Session | None = None) -> IngestionManifest:
    ensure_output_dirs()
    ingestion_timestamp = utc_now_iso()
    results: list[FundLoadResult] = []
    loaded_count = 0

    client = session or requests.Session()

    for fund in CORPUS:
        try:
            document = load_fund(
                fund,
                ingestion_timestamp=ingestion_timestamp,
                session=client,
            )
            loaded_count += 1
            results.append(
                FundLoadResult(
                    slug=fund.slug,
                    source_url=fund.source_url,
                    status="success",
                    document_path=str(
                        RAW_DOCUMENTS_DIR / f"{document.slug}.json"
                    ),
                    html_path=str(RAW_HTML_DIR / f"{document.slug}.html"),
                )
            )
        except Exception as exc:  # noqa: BLE001 - collect per-fund failures for manifest
            results.append(
                FundLoadResult(
                    slug=fund.slug,
                    source_url=fund.source_url,
                    status="failed",
                    error=str(exc),
                )
            )

    failed_count = len(CORPUS) - loaded_count
    if loaded_count == len(CORPUS):
        status = "success"
    elif loaded_count == 0:
        status = "failed"
    else:
        status = "partial"

    manifest = IngestionManifest(
        ingestion_timestamp=ingestion_timestamp,
        status=status,
        documents_loaded=loaded_count,
        documents_failed=failed_count,
        funds=results,
    )
    save_manifest(manifest)

    if status == "failed":
        raise DataLoadError(
            "All fund pages failed to load. Downstream phases are blocked (E-8)."
        )

    return manifest


def main() -> None:
    manifest = run_ingestion()
    print(
        f"Ingestion {manifest.status}: "
        f"{manifest.documents_loaded}/{len(CORPUS)} documents loaded."
    )
    print(f"Manifest: {INGESTION_MANIFEST_PATH}")
    if manifest.documents_failed:
        for fund in manifest.funds:
            if fund.status == "failed":
                print(f"  FAILED {fund.slug}: {fund.error}")


if __name__ == "__main__":
    main()
