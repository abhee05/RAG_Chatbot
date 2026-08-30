#!/usr/bin/env python3
"""Launch the Streamlit chat UI (Phase 5, src/ui/app.py).

Usage:
    .venv/bin/python scripts/run_ui.py
    → opens http://localhost:8501
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

APP_PATH = PROJECT_ROOT / "src" / "ui" / "app.py"


def main() -> None:
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.port=8501",
        "--server.headless=false",
    ]
    subprocess.run(command, check=False)


if __name__ == "__main__":
    main()