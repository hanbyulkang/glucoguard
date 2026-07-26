"""Regenerate results.md from everything on disk, without retraining.

Merges the round-1 sweep with any checkpoints trained afterwards, so the table
covers every model that exists rather than only the ones from one script run.

Usage:  python -m scripts.write_results
"""
from __future__ import annotations

import json
import sys

from src.config import ARTIFACTS_DIR
from scripts.run_sweep import write_results_md


def collect() -> dict:
    sweep_path = ARTIFACTS_DIR / "sweep.json"
    if not sweep_path.exists():
        sys.exit(f"{sweep_path} not found, run `python -m scripts.run_sweep` first.")
    payload = json.loads(sweep_path.read_text())

    known = {r["name"] for r in payload["results"]}
    extra = []
    for path in sorted(ARTIFACTS_DIR.glob("*.metrics.json")):
        blob = json.loads(path.read_text())
        name = blob.get("name")
        if not name or name in known or name.startswith(("smoke", "_")):
            continue
        cfg = blob.get("config", {})
        extra.append({
            "name": name,
            "val": blob["val"],
            "test": blob["test"],
            "n_params": blob.get("n_params", 0),
            "probabilistic": bool(cfg.get("probabilistic")),
            "classify": bool(cfg.get("classify")),
        })

    payload["results"] = payload["results"] + extra
    payload["has_risk_heads"] = any(r.get("probabilistic") or r.get("classify")
                                    for r in extra)
    return payload


def main() -> None:
    write_results_md(collect())
    print(f"Rewrote {ARTIFACTS_DIR.parent / 'results.md'}")


if __name__ == "__main__":
    main()
