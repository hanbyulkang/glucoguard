"""Compare every alarm at the *same achieved* false-alarm rate.

An earlier version of this write-up labelled results by the false-alarm budget
the threshold was tuned for on validation. That is not what the alarm delivers:
a cutoff tuned for 6 a day lands at 14.7 a day on the test patients, and the
overshoot differs per model and per population. Reading a budget as if it were
an achieved rate silently compares different operating points.

This reads recall off each model's precision-recall curve at fixed achieved
rates, which is the only comparison where the numbers sit side by side honestly.

Usage:  python -m scripts.matched_comparison
"""
from __future__ import annotations

import json

import numpy as np

from src.config import ARTIFACTS_DIR

RATES = (3.0, 8.0, 15.0)


def recall_at(curve: dict, target: float) -> float:
    fa = np.asarray(curve["false_alarms_per_day"], dtype=float)
    rec = np.asarray(curve["recall"], dtype=float)
    order = np.argsort(fa)
    return float(np.interp(target, fa[order], rec[order]))


def main() -> None:
    alarm = json.loads((ARTIFACTS_DIR / "alarm.json").read_text())
    rows = {n: {r: recall_at(v["pr_curve_test"], r) for r in RATES}
            for n, v in alarm.items()}
    order = sorted(rows, key=lambda n: -rows[n][RATES[-1]])

    print("TEST SPLIT, recall at matched achieved false-alarm rates")
    print(f"{'model':<24s}" + "".join(f"{f'{r:g} FA/day':>13s}" for r in RATES))
    for n in order:
        print(f"{n:<24s}" + "".join(f"{rows[n][r]:12.1%} " for r in RATES))

    ext_path = ARTIFACTS_DIR / "external_pr.json"
    ext_rows = {}
    if ext_path.exists():
        ext = json.loads(ext_path.read_text())
        ext_rows = {k: {r: recall_at(c, r) for r in RATES} for k, c in ext.items()}
        print("\nEXTERNAL POPULATION, same rates")
        print(f"{'':<24s}" + "".join(f"{f'{r:g} FA/day':>13s}" for r in RATES))
        for k, v in ext_rows.items():
            print(f"{k:<24s}" + "".join(f"{v[r]:12.1%} " for r in RATES))

    payload = {"rates": list(RATES), "test": rows, "external": ext_rows}
    with open(ARTIFACTS_DIR / "matched.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {ARTIFACTS_DIR / 'matched.json'}")


if __name__ == "__main__":
    main()
