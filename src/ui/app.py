"""Phase 5 — Streamlit chat UI for the HDFC Fund FAQ Assistant (FR-19–FR-23).

Run from the project root:
    streamlit run src/ui/app.py
or:
    .venv/bin/python scripts/run_ui.py

Groww-inspired palette (FR-23), welcome line (FR-19), 3 example questions
(FR-20), persistent "Facts-only. No investment advice." note (FR-21),
session-scoped chat history (FR-22, NFR-2) and a per-answer source citation
plus "Last updated from sources" line (FR-12, FR-13).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config.corpus import CORPUS
from src.ingestion.embedder import Embedder
from src.ingestion.store import VectorStoreError, get_collection_ingestion_timestamp
from src.retrieval.answer import answer_question

GROWW_GREEN = "#00D09C"
GROWW_GREEN_DARK = "#00B386"
GROWW_INK = "#0e1f1c"
GROWW_BG = "#f7fcfb"
WELCOME_TEXT = (
    "Ask factual questions about five HDFC mutual funds on Groww — expense "
    "ratios, lock-ins, minimum SIP, exit load, riskometer, and more."
)
DISCLAIMER = "Facts-only. No investment advice."

EXAMPLE_QUESTIONS = (
    "What is the expense ratio of HDFC ELSS Tax Saver Fund?",
    "What is the minimum SIP for HDFC Large Cap Fund?",
    "What is the lock-in period for HDFC ELSS?",
)


def _use_llm() -> bool:
    """LLM answers are on by default; set FAQ_LLM=0 for retrieval-only preview."""
    return os.environ.get("FAQ_LLM", "1") != "0"


def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
          .stApp {{ background-color: {GROWW_BG}; }}
          [data-testid="stHeader"] {{ background: rgba(247,252,251,0.95); }}
          [data-testid="stSidebar"] {{
            background-color: #ffffff;
            border-right: 1px solid #d9f2ec;
          }}
          h1 {{ color: {GROWW_INK}; }}
          h2, h3 {{ color: {GROWW_INK}; }}
          .groww-hero {{
            color: {GROWW_INK};
            font-size: 1.05rem;
            line-height: 1.5;
            margin-bottom: 0.75rem;
          }}
          .groww-disclaimer {{
            background: #e9fbf6;
            border: 1px solid {GROWW_GREEN};
            color: #025b4a;
            border-radius: 10px;
            padding: 0.6rem 0.9rem;
            font-size: 0.92rem;
            margin: 0.75rem 0;
          }}
          .groww-funds {{ color: #3f4b48; font-size: 0.92rem; margin-bottom: 0.4rem; }}
          .stButton > button {{
            background-color: {GROWW_GREEN};
            color: #ffffff;
            border: none;
            border-radius: 9px;
            font-weight: 600;
          }}
          .stButton > button:hover {{
            background-color: {GROWW_GREEN_DARK};
            color: #ffffff;
            border: none;
          }}
          [data-testid="stChatMessage"] {{
            background: #ffffff;
            border: 1px solid #d9f2ec;
            border-radius: 12px;
            padding: 0.5rem 0.75rem;
            margin-bottom: 0.5rem;
          }}
          [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
            background: #e6faf3;
            border-color: {GROWW_GREEN};
          }}
          [data-testid="stChatInput"] textarea {{ border-color: #b7e6dc; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _last_updated() -> str | None:
    try:
        return get_collection_ingestion_timestamp()
    except VectorStoreError:
        return None


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### HDFC Fund FAQ Assistant")
        st.markdown(
            "<div class='groww-funds'>Answers are grounded only in public "
            "Groww fund pages (5 HDFC funds).</div>",
            unsafe_allow_html=True,
        )
        for fund in CORPUS:
            st.markdown(
                f"<div class='groww-funds'>• {fund.category} — {fund.fund_name}</div>",
                unsafe_allow_html=True,
            )

        updated = _last_updated()
        if updated:
            st.markdown(
                f"<div class='groww-disclaimer'>Last updated from sources: "
                f"{updated}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='groww-disclaimer'>Vector store not ingested yet — "
                "run Phase 4 before querying (E-8).</div>",
                unsafe_allow_html=True,
            )

        st.markdown(f"<div class='groww-disclaimer'>{DISCLAIMER}</div>", unsafe_allow_html=True)
        if st.button("Clear chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()


def _example_buttons() -> str | None:
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    clicked: str | None = None
    for col, question in zip(cols, EXAMPLE_QUESTIONS, strict=True):
        if col.button(question, use_container_width=True):
            clicked = question
    return clicked


def _call_bot(question: str) -> str:
    try:
        result = answer_question(
            question,
            embedder=get_embedder(),
            call_llm=_use_llm(),
        )
        return result.message
    except VectorStoreError as exc:
        return (
            "The vector store is unavailable (E-8). Please re-run Phase 4 "
            f"ingestion first. ({exc})"
        )


@st.cache_resource(show_spinner=False)
def get_embedder() -> Embedder:
    """Share one MiniLM encoder across reruns (FR-7, latency NFR-6)."""
    return Embedder()


def main() -> None:
    st.set_page_config(
        page_title="HDFC Fund FAQ Assistant",
        page_icon=None,
        layout="centered",
    )
    _inject_css()
    _render_sidebar()

    st.title("HDFC Mutual Fund FAQ Assistant")
    st.markdown(
        f"<div class='groww-hero'>{WELCOME_TEXT}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='groww-disclaimer'>{DISCLAIMER}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("**Try one of these:**")
    pending = _example_buttons()

    st.session_state.setdefault("messages", [])
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask about an HDFC fund — e.g. expense ratio, lock-in, minimum SIP, exit load..."
    )
    if pending:
        question = pending

    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.spinner("Searching my sources..."):
            reply = _call_bot(question)
        st.session_state["messages"].append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)


if __name__ == "__main__":
    main()