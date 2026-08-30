"""Phase 5 — LLM (Mistral API) configuration."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()  # reads MISTRAL_API_KEY (and others) from the project .env file

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL_NAME = "mistral-small-latest"
MISTRAL_API_KEY_ENV = "MISTRAL_API_KEY"
MISTRAL_TIMEOUT_SECONDS = 30
MISTRAL_MAX_TOKENS = 250
MISTRAL_TEMPERATURE = 0.0


class LLMConfigError(Exception):
    """Raised when the Mistral API key is missing or misconfigured."""


def mistral_api_key() -> str:
    key = os.environ.get(MISTRAL_API_KEY_ENV, "").strip()
    if not key:
        raise LLMConfigError(
            f"{MISTRAL_API_KEY_ENV} is not set. "
            "Export it to enable the Mistral answer path (FR-10)."
        )
    return key