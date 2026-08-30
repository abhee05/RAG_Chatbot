"""Tests for Phase 5 Streamlit UI (src/ui/app.py)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "src" / "ui" / "app.py"

DISCLAIMER = "Facts-only. No investment advice."


@pytest.fixture
def app_test() -> AppTest:
    return AppTest.from_file(str(APP_PATH), default_timeout=60)


def test_ui_renders_welcome_disclaimer_and_examples(app_test):
    app_test.run()
    assert not app_test.exception

    titles = [element.value for element in app_test.title]
    assert any("HDFC" in value for value in titles)

    markdown = "\n".join(element.value for element in app_test.markdown)
    assert "Ask factual questions about five HDFC mutual funds" in markdown
    assert DISCLAIMER in markdown
    assert "Last updated from sources" in markdown

    # FR-20: exactly 3 clickable example questions.
    button_labels = [button.label for button in app_test.button]
    assert len(button_labels) >= 3
    assert any("expense ratio" in label.lower() for label in button_labels)
    assert any("SIP" in label for label in button_labels)
    assert any("lock-in" in label.lower() for label in button_labels)


def test_ui_render_survives_missing_store(app_test, monkeypatch):
    from src.ingestion import store

    def _no_collection(*args, **kwargs):
        raise store.VectorStoreError("collection not found")

    monkeypatch.setattr(store, "get_collection_ingestion_timestamp", _no_collection)
    app_test.run()
    assert not app_test.exception
    assert "not ingested" in "\n".join(
        element.value for element in app_test.markdown
    ).lower()


def test_ui_chat_replies_without_llm(app_test, monkeypatch):
    monkeypatch.setenv("FAQ_LLM", "0")  # retrieval-only: no Mistral call
    app_test.run()

    app_test.chat_input[0].set_value(
        "What is the minimum SIP for HDFC Large Cap Fund?"
    ).run()

    assert not app_test.exception
    markdown = "\n".join(element.value for element in app_test.markdown)
    assert "Retrieval-only mode" in markdown
    assert "hdfc-large-cap-fund-direct-growth" in markdown


def test_ui_chat_blocks_advice_without_llm(app_test, monkeypatch):
    monkeypatch.setenv("FAQ_LLM", "0")
    app_test.run()

    app_test.chat_input[0].set_value(
        "Should I invest in HDFC Small Cap Fund?"
    ).run()

    assert not app_test.exception
    markdown = "\n".join(element.value for element in app_test.markdown)
    assert "not investment advice" in markdown