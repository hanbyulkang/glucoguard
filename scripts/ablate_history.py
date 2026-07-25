"""How much CGM history does the model actually need?

The main sweep uses two hours of context because that is the common choice in
the literature, not because we measured it. This runs the same architecture over
several history lengths so the choice is evidence rather than convention.

Longer context is not free: it discards more windows (a window is only usable if
its entire history is intact), so the training set shrinks as the context grows.
That trade-off is part of what this measures.

Usage:  python -m scripts.ablate_history --model tcn
"""
from __future__ import annotations

import argparse
import json

from src.config import ARTIFACTS_DIR, SAMPLE_MINUTES
from src.data.windows import build_windows
from src.metrics import HEADER, format_row
from src.train import TrainConfig, run

LENGTHS = [12, 24, 36, 48]      # 1 h, 2 h, 3 h, 4 h


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="tcn", choices=["lstm", "tcn", "transformer"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps-per-epoch", type=int, default=1000)
    args = p.parse_args()

    rows = []
    for steps in LENGTHS:
        minutes = steps * SAMPLE_MINUTES
        print(f"\n{'=' * 100}\nHISTORY = {minutes} min ({steps} samples)\n{'=' * 100}",
              flush=True)
        windows = build_windows(verbose=True, history_steps=steps)
        res = run(
            TrainConfig(model=args.model, epochs=args.epochs,
                        steps_per_epoch=args.steps_per_epoch,
                        tag=f"{args.model}_hist{minutes}"),
            windows=windows,
        )
        rows.append({
            "history_minutes": minutes,
            "history_steps": steps,
            "train_windows": int(len(windows["train"])),
            "val": res["val"],
            "test": res["test"],
        })

    with open(ARTIFACTS_DIR / "ablation_history.json", "w") as fh:
        json.dump(rows, fh, indent=2)

    print(f"\n{'=' * 100}\nHISTORY LENGTH ABLATION ({args.model})\n{'=' * 100}")
    print(f"{'history':>9s} {'train windows':>14s}  {HEADER}")
    for r in rows:
        prefix = f"{r['history_minutes']:>6d}min {r['train_windows']:>14,}  "
        print(prefix + format_row("", r["test"]))

    best = min(rows, key=lambda r: r["val"]["rmse"])
    print(f"\nBest on validation: {best['history_minutes']} min of history")


if __name__ == "__main__":
    main()
