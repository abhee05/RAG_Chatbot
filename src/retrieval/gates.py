"""Phase 5 — Pre-retrieval gates (FR-15, FR-16, FR-17, E-1, E-2, E-5, E-6).

Out-of-scope queries (opinion, returns comparison, PII, empty/gibberish) are
blocked before they reach vector search (architecture.md §7.1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Open item #4 in architecture.md §13: single consistent educational URL.
GROWW_EDUCATIONAL_URL = (
    "https://groww.in/blog/all-you-need-to-know-about-mutual-funds"
)
GROWW_FUNDS_URL = "https://groww.in/mutual-funds"

PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("PAN", re.compile(r"(?<![A-Z0-9])[A-Z]{5}[0-9]{4}[A-Z](?![A-Z0-9])")),
    ("Aadhaar", re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")),
    ("phone", re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)")),
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z.]+")),
    ("card", re.compile(r"(?<!\d)(?:\d[\s-]?){13,16}(?!\d)")),
)

ADVICE_PATTERNS = re.compile(
    r"\b(should i|should we|buy|sell|invest(?: now)?|recommend|advice|"
    r"portfolio|allocate|switch|redeem all|is it good|worth it|better to)\b",
    re.IGNORECASE,
)

RETURNS_COMPARISON_PATTERNS = re.compile(
    r"\b(compare|comparison|compared|better|best|outperform(?:ed)?|vs\.?|versus)\b"
    r".{0,40}\b(returns?|perform(?:ance)?s?|profit|gains?)\b|"
    r"\b(returns?|performance|profit|gains?)\b.{0,40}"
    r"\b(compare|comparison|better|best|vs\.?|versus)\b|"
    r"\bwhich fund\b.{0,25}\b(better|best)\b",
    re.IGNORECASE,
)

GIBBERISH_PATTERNS = re.compile(
    r"(?i)\b(asdf|qwer|zxcv|jklm|asdk|fjdk|hjk|gfdsa)\w*\b"
)

NO_ADVICE_DISCLAIMER = "Facts-only. No investment advice."


@dataclass(frozen=True)
class GateResult:
    allowed: bool
    message: str | None = None
    reason: str | None = None


def contains_pii(text: str) -> str | None:
    """Return the PII type found in the input, else None (E-5)."""
    for label, pattern in PII_PATTERNS:
        if pattern.search(text):
            return label
    return None


def is_advice_request(text: str) -> bool:
    """Returns True for opinion/advice/portfolio questions (E-1)."""
    return bool(ADVICE_PATTERNS.search(text))


def is_returns_comparison(text: str) -> bool:
    """Returns True for returns/performance comparison requests (E-2)."""
    return bool(RETURNS_COMPARISON_PATTERNS.search(text))


def is_empty_or_gibberish(text: str) -> bool:
    """Returns True for empty or gibberish input (E-6)."""
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) < 3:
        return True
    letters = [char for char in stripped if char.isalpha()]
    if not letters:
        return True
    if len(set(letter.lower() for letter in letters)) == 1:
        return True
    return bool(GIBBERISH_PATTERNS.search(stripped))


def _pii_message(pii_type: str) -> str:
    return (
        "Please don't share personal information (such as a PAN, Aadhaar, "
        "phone, email, or account number). I can answer general factual "
        "questions about HDFC funds from public sources only."
    )


def _advice_message() -> str:
    return (
        "I can only share factual information from official fund pages, not "
        f"investment advice. For education on evaluating funds, see {GROWW_EDUCATIONAL_URL}\n"
        f"{NO_ADVICE_DISCLAIMER}"
    )


def _returns_message() -> str:
    return (
        "I don't compute or compare returns. For official performance data, "
        f"see the fund page fact sheet on Groww: {GROWW_FUNDS_URL}\n"
        f"{NO_ADVICE_DISCLAIMER}"
    )


def _invalid_message() -> str:
    return (
        "Please ask a factual question about one of the five HDFC funds — for "
        "example, expense ratio, lock-in period, minimum SIP, exit load, or riskometer."
    )


def evaluate_gates(text: str) -> GateResult:
    """Run pre-retrieval gates in priority order (E-5 > E-1 > E-2 > E-6)."""
    pii_type = contains_pii(text)
    if pii_type:
        return GateResult(False, _pii_message(pii_type), "pii")
    if is_advice_request(text):
        return GateResult(False, _advice_message(), "advice")
    if is_returns_comparison(text):
        return GateResult(False, _returns_message(), "returns")
    if is_empty_or_gibberish(text):
        return GateResult(False, _invalid_message(), "invalid")
    return GateResult(True)