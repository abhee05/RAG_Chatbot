# Architecture: HDFC Mutual Fund FAQ Assistant (RAG Prototype)

**Version:** 1.0  
**Status:** Draft  
**Role:** Senior Architect  
**Source of truth:** [PRD-RAG-Chatbot.md](./PRD-RAG-Chatbot.md)  
**Last updated:** August 30, 2026  

---

## 1. Purpose

This document defines a **simple, phased architecture** for the RAG prototype described in the PRD. Scope is limited to:

- Five Groww HDFC fund URLs (corpus only)
- Offline ingestion pipeline: load → chunk → embed → store
- Online query path: retrieve → prompt → Mistral answer
- Streamlit UI and edge-case handling per PRD

No production scaling, auth, or corpus expansion beyond the PRD.

---

## 2. Context Diagram

```
                    ┌─────────────────────────────────────────┐
                    │           INGESTION (offline)            │
                    │  Phase 1 → 2 → 3 → 4                    │
                    └─────────────────────────────────────────┘
  5 Groww URLs ──► Load ──► Chunk ──► Embed ──► ChromaDB
                                                    │
                    ┌─────────────────────────────────────────┐
                    │            QUERY (online)                │
                    │  Phase 5 → Mistral API → Streamlit UI   │
                    └─────────────────────────────────────────┘
  User question ──► Retrieve ──► RAG prompt ──► Answer + citation
                         ▲
                         │
              Phase 6: Retrieval testing (validates Phases 1–5)
```

### Component map (PRD stack)

| Component | Technology | PRD ref |
|-----------|------------|---------|
| Source | 5 Groww fund pages | §3 |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | FR-4, FR-7 |
| Vector store | ChromaDB | FR-5, FR-8 |
| LLM | Mistral API | FR-10 |
| UI | Streamlit (Groww-inspired colors) | FR-19–FR-23 |

---

## 3. Phase 1 — Data Loading

**Goal:** Fetch and normalize readable text from exactly five public Groww URLs.  
**PRD requirements:** FR-1, FR-2, FR-6, NFR-1  

### 3.1 Inputs

| Field | Value |
|-------|-------|
| URLs | Large-cap, Flexi-cap, ELSS, Small-cap, Hybrid (see PRD §3) |
| Allowed sources | groww.in fund pages only |
| Forbidden | Third-party blogs, app back-end screenshots, pages outside the 5 URLs |

### 3.2 Process

```
FOR each URL in corpus:
  1. HTTP fetch HTML
  2. Parse DOM → extract readable text
  3. Capture fund metadata: fund_name, category, source_url
  4. Persist raw snapshot (mitigates Groww layout changes — PRD §11)
  5. Record ingestion_timestamp (for “Last updated from sources” — FR-6)
```

### 3.3 Output contract

```json
{
  "fund_name": "HDFC ELSS Tax Saver Fund Direct Plan Growth",
  "category": "ELSS",
  "source_url": "https://groww.in/mutual-funds/...",
  "raw_text": "...",
  "ingestion_timestamp": "2026-08-30T00:00:00Z"
}
```

### 3.4 Exit criteria

- All 5 URLs load successfully (PRD success metric: 100% ingestion)
- Extracted text includes factual fields cited in PRD: expense ratio, loads, lock-in, riskometer, benchmark, SIP minimums, etc.
- On scrape failure: block downstream answers; surface error state (E-8)

---

## 4. Phase 2 — Chunking

**Goal:** Split loaded documents into retrieval-friendly segments for short factual Q&A.  
**PRD requirements:** FR-3  

### 4.1 Strategy

- Chunk **per fund document** (preserve `source_url` and `fund_name` on every chunk)
- Use overlap suitable for factual Q&A (exact size tunable in prototype; not specified in PRD)
- Prefer semantic boundaries (sections, tables, labeled fields) over arbitrary splits where page structure allows

### 4.2 Process

```
FOR each loaded document:
  1. Split raw_text into chunks with overlap
  2. Attach metadata to each chunk:
       - chunk_id (unique)
       - fund_name
       - source_url
       - ingestion_timestamp
  3. Emit chunk list for embedding phase
```

### 4.3 Output contract

```json
{
  "chunk_id": "hdfc-elss-003",
  "text": "Lock-in period: 3 years ...",
  "fund_name": "HDFC ELSS Tax Saver Fund Direct Plan Growth",
  "source_url": "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
  "ingestion_timestamp": "2026-08-30T00:00:00Z"
}
```

### 4.4 Exit criteria

- Every chunk traceable to exactly one of the 5 source URLs
- No cross-URL or external content in chunks

---

## 5. Phase 3 — Embedding

**Goal:** Convert chunks and (later) user questions into vectors using one shared model.  
**PRD requirements:** FR-4, FR-7, NFR-5  

### 5.1 Model

| Property | Value |
|----------|-------|
| Model | `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face) |
| Usage | Ingestion (batch) and query-time (single question) |
| Constraint | Same model at ingest and retrieve (FR-7) |

### 5.2 Process

```
FOR each chunk from Phase 2:
  1. Encode chunk.text → embedding vector (384-dim for MiniLM-L6-v2)
  2. Pass vector + metadata to Phase 4

At query time (Phase 5):
  1. Encode user question with identical model + tokenizer
  2. Use vector for similarity search
```

### 5.3 Exit criteria

- All chunks embedded without model mismatch between ingest and query paths
- Lightweight enough for local/dev prototype (NFR-5)

---

## 6. Phase 4 — Vector Store

**Goal:** Persist embeddings and metadata for similarity search.  
**PRD requirements:** FR-5, FR-6, US-5  

### 6.1 Store

| Property | Value |
|----------|-------|
| Engine | ChromaDB |
| Collection | Single collection for all 5 funds |
| Stored per record | embedding, chunk text, chunk_id, fund_name, source_url, ingestion_timestamp |

### 6.2 Process

```
1. Initialize / open ChromaDB collection
2. Upsert all chunk embeddings + metadata from Phase 3
3. Support full re-ingestion (US-5): replace collection on re-run
4. Expose ingestion_timestamp at collection level for UI display (FR-13)
```

### 6.3 Data layout

```
ChromaDB Collection: hdfc_fund_faq
├── id: chunk_id
├── embedding: float[]
├── document: chunk text
└── metadata:
    ├── fund_name
    ├── source_url
    └── ingestion_timestamp
```

### 6.4 Exit criteria

- All ingested chunks searchable by vector similarity
- Re-ingestion refreshes corpus and timestamp without manual DB edits

---

## 7. Phase 5 — Retrieval Logic

**Goal:** Given a user question, return the best grounded context for Mistral answer generation.  
**PRD requirements:** FR-7–FR-13, FR-14–FR-18  

### 7.1 Query pipeline

```
User input (Streamlit)
        │
        ▼
┌───────────────────┐
│ Pre-retrieval     │  FR-15, FR-16, FR-17
│ gates             │  • PII → reject (E-5)
│                   │  • Opinion/advice → refuse + educational link (E-1)
│                   │  • Returns comparison → refuse + factsheet link (E-2)
│                   │  • Empty/gibberish → reprompt (E-6)
└─────────┬─────────┘
          │ factual query
          ▼
┌───────────────────┐
│ Embed question    │  FR-7 (same MiniLM model)
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Vector search     │  FR-8 (top-k from ChromaDB)
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Relevance check   │  FR-18: no match → “not in my sources” (E-3)
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ RAG prompt build  │  FR-9: retrieved chunks + system rules
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Mistral API       │  FR-10: ≤3 sentences (FR-11)
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Post-process      │  FR-12: one citation (best source_url)
│                   │  FR-13: “Last updated from sources: <date>”
└─────────┬─────────┘
          ▼
    Streamlit response
```

### 7.2 Retrieval rules

| Rule | PRD basis |
|------|-----------|
| top-k retrieval from ChromaDB | FR-8 (k value: open question in PRD §13) |
| Select **one** citation URL for answer | FR-12, E-7 |
| Answer only from retrieved context | NFR-4, PRD §11 (anti-hallucination) |
| No performance computation | NFR-3, FR-16 |
| Ambiguous fund name | Clarify or answer with caveat (E-4) |
| Mistral failure | User-friendly retry; no fabricated facts (E-9) |

### 7.3 Prompt contract (Mistral)

System instructions must enforce:

- Facts-only; no investment advice
- Answer strictly from provided chunks
- Maximum 3 sentences
- Do not compute or compare returns

User prompt includes: question + retrieved chunk texts + metadata (fund_name, source_url).

### 7.4 UI integration (Streamlit)

Retrieval logic is invoked from Streamlit chat (FR-22). UI also renders:

- Welcome line, 3 example questions, disclaimer (FR-19–FR-21)
- Groww-inspired color palette (FR-23)
- Session-scoped history only — no PII persistence (NFR-2)

### 7.5 Exit criteria

- Factual questions return answer + single citation + last-updated line
- Out-of-scope queries never reach vector search when gates apply
- End-to-end latency target: ~10s (NFR-6)

---

## 8. Phase 6 — Retrieval Testing

**Goal:** Validate ingestion quality, retrieval accuracy, and query-path behavior before demo.  
**PRD requirements:** §9 Edge Cases, §10 Success Metrics, M5/M6 milestones  

### 8.1 Test layers

```
Layer A: Ingestion tests     → Phases 1–4 (all 5 URLs ingested)
Layer B: Retrieval tests     → Phase 5 (correct chunks for factual Qs)
Layer C: End-to-end tests    → Full RAG + UI + refusal paths
```

### 8.2 Retrieval test cases (factual)

| ID | Query (example) | Expected retrieval source |
|----|-----------------|---------------------------|
| RT-1 | Expense ratio of HDFC ELSS? | ELSS fund page chunks |
| RT-2 | ELSS lock-in period? | ELSS fund page chunks |
| RT-3 | Minimum SIP for HDFC Large Cap? | Large-cap page chunks |
| RT-4 | Exit load on HDFC Small Cap? | Small-cap page chunks |
| RT-5 | Riskometer / benchmark for Balanced Advantage? | Hybrid page chunks |
| RT-6 | Capital gains statement download? | Relevant page chunk or honest gap (E-10) |

**Pass criteria:** Retrieved chunks contain the fact needed to answer; cited URL matches fund asked about.

### 8.3 Edge-case test matrix (PRD §9)

| ID | Scenario | Expected | Blocks retrieval? |
|----|----------|----------|-------------------|
| E-1 | “Should I buy?” | Refusal + educational link | Yes |
| E-2 | Returns comparison | No computation; factsheet link | Yes |
| E-3 | Fund not in corpus | “Not in my sources” | N/A (no valid chunk) |
| E-4 | Ambiguous fund | Clarify or caveat | Maybe |
| E-5 | PII in input | Reject; no echo/store | Yes |
| E-6 | Empty / gibberish | Reprompt | Yes |
| E-7 | Multi-page answer | Single best citation | No |
| E-8 | Scrape / stale corpus | Error state; block answers | Yes |
| E-9 | Mistral timeout | Retry message | N/A |
| E-10 | Capital gains how-to | Factual or honest gap | No |

### 8.4 Success metrics (PRD §10)

| Metric | Target |
|--------|--------|
| Factual Q&A accuracy (n=20 spot-check) | ≥ 85% grounded in cited page |
| Citation on every answered query | 100% |
| Opinion/advice correctly refused | 100% |
| PII inputs blocked | 100% |
| All 5 URLs ingested | 100% |
| Edge-case suite pass rate | ≥ 90% |

### 8.5 Exit criteria

- Layer A–C tests documented and repeatable
- Prototype ready for manual QA + demo (M6)

---

## 9. Phase Dependencies

```
Phase 1 (Data Loading)
    └──► Phase 2 (Chunking)
            └──► Phase 3 (Embedding)
                    └──► Phase 4 (Vector Store)
                            └──► Phase 5 (Retrieval Logic)
                                    └──► Phase 6 (Retrieval Testing)
```

Phases 1–4 run **offline** (ingestion job). Phase 5 runs **online** per user query. Phase 6 validates the full chain.

---

## 10. Project Folder Structure

```
BuildClass/
├── docs/                          # PRD, architecture, problem statement
│
├── data/                          # All pipeline artifacts (generated at runtime)
│   ├── raw/
│   │   ├── html/                  # Phase 1: HTML snapshots per fund URL
│   │   └── documents/             # Phase 1: parsed JSON (fund_name, raw_text, etc.)
│   ├── chunks/                    # Phase 2: chunked JSON with metadata
│   ├── embeddings/                # Phase 3: optional embedding cache / exports
│   ├── vectordb/                  # Phase 4: ChromaDB persist directory
│   └── metadata/                  # Ingestion manifest, timestamps (FR-6)
│
├── src/                           # Application code
│   ├── config/                    # Corpus URLs, model names, paths
│   ├── ingestion/                 # Phases 1–4: load, chunk, embed, store
│   ├── retrieval/                 # Phase 5: gates, retrieve, prompt, LLM
│   └── ui/                        # Streamlit app (FR-19–FR-23)
│
└── tests/                         # Phase 6: retrieval & edge-case tests
    ├── retrieval/                 # RT-1–RT-6 factual retrieval tests
    └── edge_cases/                # E-1–E-10 refusal & error-path tests
```

### Data ↔ phase mapping

| Folder | Phase | Contents |
|--------|-------|----------|
| `data/raw/html/` | 1 | Raw HTML snapshots (Groww layout change mitigation) |
| `data/raw/documents/` | 1 | Parsed document JSON per fund |
| `data/chunks/` | 2 | Chunk files with `chunk_id`, `source_url`, `fund_name` |
| `data/embeddings/` | 3 | Intermediate embedding artifacts (pre-ChromaDB) |
| `data/vectordb/` | 4 | ChromaDB collection (`hdfc_fund_faq`) |
| `data/metadata/` | 1, 4 | `ingestion_timestamp`, corpus manifest |

### Code ↔ phase mapping

| Module | Phase | Responsibility |
|--------|-------|----------------|
| `src/config/` | All | 5 fund URLs, paths, model/API settings |
| `src/ingestion/loader.py` | 1 | Fetch & parse Groww pages |
| `src/ingestion/chunker.py` | 2 | Split documents into chunks |
| `src/ingestion/embedder.py` | 3 | MiniLM-L6-v2 encoding |
| `src/ingestion/store.py` | 4 | ChromaDB upsert & re-ingestion |
| `src/retrieval/gates.py` | 5 | PII, advice, returns pre-checks |
| `src/retrieval/retriever.py` | 5 | Vector search (top-k) |
| `src/retrieval/prompt.py` | 5 | RAG prompt assembly |
| `src/retrieval/llm.py` | 5 | Mistral API call |
| `src/ui/app.py` | 5 | Streamlit chat UI |
| `tests/retrieval/` | 6 | Factual Q&A retrieval tests |
| `tests/edge_cases/` | 6 | Refusal, PII, error-path tests |

Generated data artifacts are gitignored; folder structure is kept via `.gitkeep` files.

---

## 11. PRD Requirement Traceability

| Phase | PRD requirements |
|-------|------------------|
| 1 — Data Loading | FR-1, FR-2, FR-6, NFR-1 |
| 2 — Chunking | FR-3 |
| 3 — Embedding | FR-4, FR-7, NFR-5 |
| 4 — Vector Store | FR-5, FR-6, US-5 |
| 5 — Retrieval Logic | FR-7–FR-18, FR-19–FR-23, NFR-2–NFR-6 |
| 6 — Retrieval Testing | §9, §10, M5, M6 |

---

## 12. Out of Scope (per PRD)

- Investment advice, portfolio recommendations, buy/sell guidance
- Return calculations and performance comparisons
- User accounts, auth, PII storage
- Corpus beyond 5 URLs
- Production monitoring, SLA, horizontal scaling

---

## 13. Open Items (from PRD §13)

Carried forward unchanged — to be decided during implementation, not expanded in this architecture:

1. Disambiguation UI for ambiguous fund names (E-4)
2. Fixed vs dynamic top-k for retrieval
3. Manual vs scheduled re-ingestion
4. Standard Groww educational URL for refusal messages
