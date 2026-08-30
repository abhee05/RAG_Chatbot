#!/usr/bin/env python3
"""Run Phase 2 chunking for all loaded fund documents."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.chunker import main

if __name__ == "__main__":
    main()
