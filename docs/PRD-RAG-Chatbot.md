# PRD: HDFC Mutual Fund FAQ Assistant (RAG Prototype)

**Version:** 1.0  
**Status:** Draft  
**Author:** PM  
**Last updated:** August 30, 2026  

---

## 1. Overview

### 1.1 Product Summary

Build a **facts-only FAQ chatbot prototype** powered by Retrieval-Augmented Generation (RAG). The assistant answers factual questions about five HDFC mutual fund schemes using content scraped exclusively from their official Groww fund pages. This is a **hobby/learning project** to validate RAG ingestion, retrieval, and grounded answer generation—not a production investment product.

### 1.2 Problem Statement

Retail investors researching HDFC funds on Groww often need quick answers to factual questions (expense ratio, lock-in period, minimum SIP, exit load, riskometer, etc.) without wading through long fund pages or receiving unsolicited investment advice. A lightweight, citation-backed FAQ assistant can surface these facts reliably from a scoped, public corpus.

### 1.3 Target Users

| Persona | Need |
|---------|------|
| **Curious investor** | Quick factual lookup while comparing HDFC fund options |
| **Builder / learner** | Hands-on RAG pipeline experiment (ingestion → retrieval → LLM) |

---

## 2. Goals & Non-Goals

### 2.1 Goals

- Answer **factual queries only** with **one clear citation link** per response.
- Ground every answer in content from the **five specified Groww URLs** (HDFC AMC funds only).
- Deliver a **minimal Streamlit UI** styled with Groww-inspired colors.
- Implement a complete RAG stack: scrape → chunk → embed → store (ChromaDB) → retrieve → generate (Mistral API).
- Include **edge-case testing** for refusal, missing data, and constraint violations.

### 2.2 Non-Goals

- Investment advice, portfolio recommendations, or buy/sell guidance.
- Return calculations, performance comparisons, or predictive claims.
- User accounts, authentication, or conversation persistence with PII.
- Corpus expansion beyond the five URLs.
- Production-grade scalability, monitoring, or SLA guarantees.

---

## 3. Corpus Scope

**Source website:** [groww.in](https://groww.in/)  
**AMC:** HDFC  

| Category | Fund | URL |
|----------|------|-----|
| Large-cap | HDFC Large Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| Flexi-cap | HDFC Flexi Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| ELSS | HDFC ELSS Tax Saver Fund Direct Plan Growth | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth |
| Small-cap | HDFC Small Cap Fund Direct Growth | https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth |
| Hybrid | HDFC Balanced Advantage Fund Direct Growth | https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth |

**Hard rule:** Only content from the URLs above. No third-party blogs, no app back-end screenshots, no other pages.

---

## 4. User Stories

| ID | As a… | I want to… | So that… |
|----|-------|------------|----------|
| US-1 | User | Ask “What is the expense ratio of HDFC ELSS?” | I get a short, cited factual answer |
| US-2 | User | Ask “What is the ELSS lock-in period?” | I know the regulatory lock-in without advice |
| US-3 | User | Ask “Should I buy HDFC Small Cap?” | The bot politely refuses and points to educational content |
| US-4 | User | See 3 example questions on load | I understand what the bot can answer |
| US-5 | Builder | Re-run ingestion on the 5 URLs | The vector store stays in sync with source pages |
| US-6 | User | See “Last updated from sources: \<date\>” | I know how fresh the answer grounding is |

---

## 5. Functional Requirements

### 5.1 Data Ingestion

| Req ID | Requirement | Priority |
|--------|-------------|----------|
| FR-1 | Fetch and parse HTML from all 5 Groww fund URLs | P0 |
| FR-2 | Extract readable text (fund name, ratios, loads, lock-in, riskometer, benchmark, SIP minimums, etc.) | P0 |
| FR-3 | Chunk documents with overlap suitable for short factual Q&A | P0 |
| FR-4 | Embed chunks using `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face) | P0 |
| FR-5 | Persist embeddings + metadata (source URL, fund name, chunk id) in **ChromaDB** | P0 |
| FR-6 | Store ingestion timestamp for “Last updated from sources” display | P1 |

### 5.2 Retrieval & Generation

| Req ID | Requirement | Priority |
|--------|-------------|----------|
| FR-7 | Embed user question with the same model used at ingestion | P0 |
| FR-8 | Retrieve top-k relevant chunks from ChromaDB | P0 |
| FR-9 | Construct LLM prompt with retrieved context + system rules | P0 |
| FR-10 | Generate answer via **Mistral API** | P0 |
| FR-11 | Limit answers to **≤ 3 sentences** | P0 |
| FR-12 | Append **exactly one citation link** (the most relevant source URL) | P0 |
| FR-13 | Append **“Last updated from sources: \<date\>”** | P0 |

### 5.3 Query Handling & Refusal

| Req ID | Requirement | Priority |
|--------|-------------|----------|
| FR-14 | Accept factual questions (expense ratio, lock-in, min SIP, exit load, riskometer, benchmark, capital-gains statement download, etc.) | P0 |
| FR-15 | Refuse opinionated/portfolio questions with a polite, facts-only message + relevant educational link | P0 |
| FR-16 | Refuse or redirect performance/returns comparison requests; link to official factsheet if applicable | P0 |
| FR-17 | Reject input containing PII (PAN, Aadhaar, account numbers, OTP, email, phone) with a clear message | P0 |
| FR-18 | Respond gracefully when retrieval finds no relevant chunk (“I don’t have that information in my sources”) | P1 |

### 5.4 UI (Streamlit)

| Req ID | Requirement | Priority |
|--------|-------------|----------|
| FR-19 | Welcome line explaining the assistant’s purpose | P0 |
| FR-20 | Display **3 clickable example questions** | P0 |
| FR-21 | Persistent note: **“Facts-only. No investment advice.”** | P0 |
| FR-22 | Chat input + message history (session-scoped) | P0 |
| FR-23 | Color palette aligned with Groww website (greens/teals, clean white background) | P1 |

---

## 6. Non-Functional Requirements

| Req ID | Requirement |
|--------|-------------|
| NFR-1 | **Public sources only** — no proprietary or third-party blog content |
| NFR-2 | **No PII storage** — do not persist PAN, Aadhaar, accounts, OTPs, emails, phones |
| NFR-3 | **No performance claims** — no computed or comparative returns |
| NFR-4 | **Transparency** — every answer cites one source; show last-updated date |
| NFR-5 | **Lightweight** — free/local embedding model; suitable for local/dev prototype |
| NFR-6 | **Latency (prototype)** — end-to-end response under ~10s on typical hardware |

---

## 7. System Architecture (High Level)

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────┐
│ 5 Groww URLs│ ──► │ Scrape/Chunk │ ──► │ MiniLM Embed│ ──► │ ChromaDB │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────┘
                                                                    │
┌─────────────┐     ┌──────────────┐     ┌─────────────┐          │
│  Streamlit  │ ◄── │ Mistral API  │ ◄── │ RAG Prompt  │ ◄────────┘
│     UI      │     │   (answer)   │     │ + retrieve  │
└─────────────┘     └──────────────┘     └─────────────┘
```

**Stack summary**

| Layer | Choice |
|-------|--------|
| UI | Streamlit |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB |
| LLM | Mistral API |
| Source | 5 Groww HDFC fund pages only |

---

## 8. Example Interactions

### 8.1 In Scope (Answer)

**User:** What is the expense ratio of HDFC ELSS Tax Saver Fund Direct Plan Growth?

**Assistant:** The expense ratio for HDFC ELSS Tax Saver Fund Direct Plan Growth is X.XX% (Direct plan).  
**Source:** https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth  
**Last updated from sources:** August 30, 2026

### 8.2 Out of Scope (Refuse)

**User:** Should I sell my HDFC Small Cap holdings now?

**Assistant:** I can only share factual information from official fund pages, not investment advice. For education on evaluating holdings, see [relevant Groww educational link].  
**Facts-only. No investment advice.**

### 8.3 PII Rejection

**User:** My PAN is ABCDE1234F, can you check my ELSS?

**Assistant:** Please don’t share personal information like PAN. I can answer general factual questions about HDFC funds from public sources only.

---

## 9. Edge Cases & Test Scenarios

| # | Scenario | Expected Behavior |
|---|----------|-------------------|
| E-1 | Opinion question (“Should I buy?”) | Polite refusal + educational link |
| E-2 | Returns comparison (“Which fund gave better returns?”) | No computation; link to factsheet/source |
| E-3 | Question about fund not in corpus (e.g., HDFC Mid Cap) | Clear “not in my sources” message |
| E-4 | Ambiguous fund (“What’s the expense ratio?”) | Ask clarifying question or answer for most likely match with caveat |
| E-5 | PII in message | Reject; do not store or echo PII |
| E-6 | Empty or gibberish input | Prompt user to ask a factual fund question |
| E-7 | Question answerable on multiple pages | Pick single best citation; keep answer ≤3 sentences |
| E-8 | Stale corpus / scrape failure | Show error state; block answers until re-ingestion succeeds |
| E-9 | Mistral API timeout/error | User-friendly retry message; no hallucinated facts |
| E-10 | Capital gains statement download how-to | Factual steps/link from source if present; otherwise honest gap |

---

## 10. Success Metrics (Prototype)

| Metric | Target |
|--------|--------|
| Factual Q&A accuracy (manual spot-check, n=20) | ≥ 85% grounded in cited page |
| Citation present on every answered query | 100% |
| Opinion/advice questions correctly refused | 100% |
| PII inputs detected and blocked | 100% |
| Ingestion completes for all 5 URLs | 100% |
| Edge-case test suite pass rate | ≥ 90% |

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Groww page structure changes break scraper | Store raw HTML snapshots; version ingestion script |
| LLM hallucination beyond retrieved chunks | Strict prompt: answer only from context; low temperature |
| User treats prototype as financial advice | Prominent disclaimer in UI and every refusal path |
| Mistral API cost/latency | Cache frequent Q&A optionally in v2; keep prototype queries minimal |

---

## 12. Milestones (Suggested)

| Phase | Deliverable | Duration |
|-------|-------------|----------|
| M1 | Scraper + chunker for 5 URLs | 2 days |
| M2 | ChromaDB ingestion pipeline + embedding | 1 day |
| M3 | RAG retrieval + Mistral prompt/answer layer | 2 days |
| M4 | Streamlit UI + Groww styling | 1 day |
| M5 | Refusal logic, PII filter, edge-case tests | 2 days |
| M6 | Manual QA + demo | 1 day |

**Total estimate:** ~9 working days for a working prototype.

---

## 13. Open Questions

1. Should ambiguous fund names trigger a disambiguation UI (dropdown of 5 funds)?
2. Fixed `top-k` for retrieval (e.g., k=3) or dynamic based on question type?
3. Re-ingestion cadence: manual only, or scheduled weekly refresh?
4. Which Groww educational URL to use consistently in refusal messages?

---

## 14. Appendix: Compliance Copy (UI)

**Welcome:** Ask factual questions about five HDFC mutual funds on Groww—expense ratios, lock-ins, minimum SIP, exit load, and more.

**Disclaimer (always visible):** Facts-only. No investment advice.

**Footer note on answers:** Last updated from sources: \<ingestion_date\>
