"""Warm the forecast cache so the demo opens instantly.

Runs every held-out patient through a checkpoint once and stores the result.
The app reads those files rather than re-running the model on each interaction.

Usage:  python -m scripts.precompute_forecasts [checkpoint ...]
"""
from __future__ import annotations

import sys
import time

from src.predictor import (
    Forecaster,
    best_checkpoint,
    cached_forecast,
    load_splits,
)


def main() -> None:
    names = sys.argv[1:] or [n for n in [best_checkpoint()] if n]
    if not names:
        sys.exit("No checkpoint found — train a model first.")

    patients = load_splits()["test"]
    for name in names:
        fc = Forecaster(name)
        for pid in patients:
            t0 = time.time()
            frame = cached_forecast(pid, fc)
            print(f"{name}/{pid}: {len(frame):>8,} rows  [{time.time() - t0:5.1f}s]",
                  flush=True)


if __name__ == "__main__":
    main()
