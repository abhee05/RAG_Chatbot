"""Phase 5 — Mistral API call (FR-10) with user-friendly error handling (E-9)."""

from __future__ import annotations

import requests

from src.config.llm import (
    MISTRAL_API_KEY_ENV,
    MISTRAL_API_URL,
    MISTRAL_MAX_TOKENS,
    MISTRAL_MODEL_NAME,
    MISTRAL_TEMPERATURE,
    MISTRAL_TIMEOUT_SECONDS,
    mistral_api_key,
)


class LLMError(Exception):
    """Raised when the Mistral API cannot produce an answer (E-9)."""


def generate_answer(
    messages: list[dict[str, str]],
    model: str = MISTRAL_MODEL_NAME,
    timeout: int = MISTRAL_TIMEOUT_SECONDS,
) -> str:
    """Send chat messages to Mistral and return the trimmed answer text."""
    try:
        api_key = mistral_api_key()
    except Exception as exc:
        raise LLMError(
            f"{MISTRAL_API_KEY_ENV} is not configured. "
            "Not calling the LLM (no hallucinated facts)."
        ) from exc

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": MISTRAL_TEMPERATURE,
        "max_tokens": MISTRAL_MAX_TOKENS,
        "stream": False,
    }

    try:
        response = requests.post(
            MISTRAL_API_URL,
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise LLMError(f"Mistral API request failed: {exc}") from exc

    if response.status_code != 200:
        raise LLMError(
            f"Mistral API returned HTTP {response.status_code}: {response.text[:300]}"
        )

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError(f"Mistral API returned an unexpected payload: {exc}") from exc

    if not content:
        raise LLMError("Mistral API returned an empty answer.")
    return content