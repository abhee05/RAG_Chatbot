#!/usr/bin/env python3
"""Phase 5 — CLI to test backend retrieval and RAG answers.

Usage:
  python scripts/ask.py "What is the expense ratio of HDFC ELSS?"
      → retrieval-only: shows retrieved chunks + cosine scores (no LLM needed)

  python scripts/ask.py --llm "What is the ELSS lock-in period?"
      → full RAG: gates → retrieve → Mistral answer + citation + last-updated
        (requires MISTRAL_API_KEY in the environment)

  python scripts/ask.py --top-k 5 "Minimum SIP for HDFC Large Cap Fund?"
      → override default top-k (default 3)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.answer import answer_question
from src.retrieval.gates import evaluate_gates
from src.retrieval.retriever import format_retrieval, retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the Phase 5 retrieval backend.")
    parser.add_argument("question", help="Question to ask the FAQ assistant")
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Call Mistral to generate the final answer (needs MISTRAL_API_KEY)",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the retrieval result as JSON",
    )
    args = parser.parse_args()

    gate = evaluate_gates(args.question)
    if not gate.allowed:
        if args.json:
            print(
                json.dumps(
                    {
                        "question": args.question,
                        "blocked": True,
                        "reason": gate.reason,
                        "message": gate.message,
                    },
                    indent=2,
                    ensure_ascii=True,
                )
            )
        else:
            print(f"[blocked:{gate.reason}] {gate.message}")
        return

    if args.json:
        result = retrieve(args.question, top_k=args.top_k)
        payload = {
            "question": args.question,
            "reason": result.reason,
            "relevant": result.relevant,
            "message": result.message,
            "best_source_url": result.best_source_url,
            "last_updated": result.last_updated,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "fund_name": c.fund_name,
                    "source_url": c.source_url,
                    "cosine_score": score,
                    "distance": c.distance,
                    "text": c.text,
                }
                for c, score in zip(result.chunks, result.cosine_scores, strict=True)
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return

    if not args.llm:
        print(format_retrieval(retrieve(args.question, top_k=args.top_k)))
        return

    answer = answer_question(args.question, top_k=args.top_k, call_llm=True)
    print(f"status: {answer.status} | reason: {answer.reason}")
    print("---")
    print(answer.message)


if __name__ == "__main__":
    main()