# HDFC Mutual Fund FAQ Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers factual questions about five HDFC mutual funds, grounded strictly in public Groww fund pages. Built as a phased RAG prototype: it ingests a fixed corpus, indexes it in a vector store, and answers queries with citations — no investment advice.

## Highlights

- **Factual & grounded** — Answers cite one source URL and a "Last updated from sources" date (no hallucination or advice).
- **RAG pipeline** — Load → chunk → embed → store (offline) then retrieve → prompt → LLM (online).
- **Pre-retrieval safety gates** — Blocks PII, opinion/advice requests, return-comparison prompts, and gibberish before they hit search.
- **Streamlit UI** — Groww-inspired chat interface with example questions and session-scoped history.

## Corpus (5 HDFC funds)

| Category | Fund |
|----------|------|
| Large-cap | HDFC Large Cap Fund Direct Growth |
| Flexi-cap | HDFC Flexi Cap Fund Direct Growth |
| ELSS | HDFC ELSS Tax Saver Fund Direct Plan Growth |
| Small-cap | HDFC Small Cap Fund Direct Growth |
| Hybrid | HDFC Balanced Advantage Fund Direct Growth |

## Architecture

```
                       INGESTION (offline)                     QUERY (online)
  5 Groww URLs ─► Load ─► Chunk ─► Embed ─► ChromaDB      User question ─► Retrieve
                                                                    │
                                  ... ─► streamlit/ui ⟵ Mistral ── RAG prompt
```

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (same model at ingest and query)
- **Vector store:** ChromaDB (`hdfc_fund_faq` collection)
- **LLM:** Mistral API (`mistral-small-latest`)
- **UI:** Streamlit

The pipeline runs in six phases — see [`docs/architecture.md`](docs/architecture.md) and [`docs/PRD-RAG-Chatbot.md`](docs/PRD-RAG-Chatbot.md) for full details.

## Project layout

```
BuildClass/
├── docs/                  # PRD, architecture, problem statement
├── data/                  # Runtime artifacts (raw, chunks, embeddings, vectordb)
├── src/
│   ├── config/            # Corpus URLs, model/API settings, paths
│   ├── ingestion/         # Phases 1–4: load, chunk, embed, store
│   ├── retrieval/         # Phase 5: gates, retrieve, prompt, LLM
│   └── ui/                # Streamlit app
├── scripts/               # CLI entry points for each phase
└── tests/                 # Phase 6: retrieval & edge-case tests
```

## Setup

Requires Python 3.10+.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create a .env file (see below)
cp .env.example .env   # or create one manually
```

### Environment variables

Create a `.env` file in the project root:

```
MISTRAL_API_KEY=your_mistral_api_key
```

`MISTRAL_API_KEY` is only required for the full LLM answer path (`--llm` / UI default). Retrieval-only mode works without it.

## Usage

### Run the Streamlit UI

```bash
.venv/bin/python scripts/run_ui.py
# or
streamlit run src/ui/app.py
```

Opens http://localhost:8501. Set `FAQ_LLM=0` for retrieval-only preview.

### Ingest the corpus (offline)

```bash
.venv/bin/python scripts/load_data.py     # Phase 1 - load
.venv/bin/python scripts/chunk_data.py    # Phase 2 - chunk
.venv/bin/python scripts/embed_data.py    # Phase 3 - embed
.venv/bin/python scripts/store_data.py    # Phase 4 - store in ChromaDB
```

### Ask from the CLI

```bash
# retrieval-only (no LLM needed): shows chunks + cosine scores
.venv/bin/python scripts/ask.py "What is the expense ratio of HDFC ELSS?"

# full RAG (needs MISTRAL_API_KEY): answer + citation + last-updated
.venv/bin/python scripts/ask.py --llm "What is the ELSS lock-in period?"

# override top-k (default 3)
.venv/bin/python scripts/ask.py --top-k 5 "Minimum SIP for HDFC Large Cap Fund?"

# JSON output
.venv/bin/python scripts/ask.py --json "What is the exit load on HDFC Small Cap?"
```

## Testing

```bash
pytest
```

Covers retrieval (factual Q&A across the corpus) and edge-case behavior (PII, advice, returns-comparison, gibberish refusals).

## Out of scope

- Investment advice, buy/sell recommendations, portfolio guidance
- Return calculations and performance comparisons
- User accounts / authentication / PII storage
- Corpus beyond the 5 fixed fund URLs

## Documentation

- [Architecture](docs/architecture.md)
- [PRD](docs/PRD-RAG-Chatbot.md)
- [Problem statement](docs/problemstatement.txt)
