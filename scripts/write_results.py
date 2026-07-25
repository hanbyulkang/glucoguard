"""Regenerate results.md from artifacts/sweep.json without retraining.

Useful when the write-up changes but the numbers have not.

Usage:  python -m scripts.write_results
"""
from __future__ import annotations

import json
import sys

from src.config import ARTIFACTS_DIR
from scripts.run_sweep import write_results_md


def main() -> None:
    path = ARTIFACTS_DIR / "sweep.json"
    if not path.exists():
        sys.exit(f"{path} not found — run `python -m scripts.run_sweep` first.")
    write_results_md(json.loads(path.read_text()))
    print(f"Rewrote {ARTIFACTS_DIR.parent / 'results.md'}")


if __name__ == "__main__":
    main()
