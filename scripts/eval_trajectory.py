"""Watch each wearer's alarm threshold move as it is re-fitted week after week.

`eval_policy.py` says the rolling policy works. This asks what it is actually
doing: does a wearer's threshold settle after a few refits and stay put, or does
it keep chasing? And when it moves, does it move because the person changed?

Every refit uses only the trailing four weeks, so a threshold in force on a
given day was fitted on data strictly earlier than that day.

Usage:  python -m scripts.eval_trajectory
"""
from __future__ import annotations

import json

import numpy as np

from src.alarm_policy import rolling_thresholds
from src.config import ARTIFACTS_DIR, HYPO_THRESHOLD
from src.data.windows import build_windows
from src.predictor import Forecaster
from scripts.eval_calibration import scores_for
from scripts.eval_external import external_windows

TARGET = 6.0


def trajectories(ws, fc: Forecaster) -> dict:
    score = scores_for(fc, ws.X)
    out = {}
    for pid in sorted(set(ws.patient_ids)):
        sel = ws.patient_ids == pid
        y, s = ws.y[sel], score[sel]
        t = np.asarray(ws.times[sel], dtype="datetime64[ns]")
        _, info = rolling_thresholds(y, s, t, TARGET)
        sched = info["schedule"]
        if len(sched) < 4:
            continue

        thr = np.array([p["threshold"] for p in sched])
        rate = np.array([p["trailing_hypo_rate"] for p in sched])
        steps = np.abs(np.diff(thr))
        finite = np.isfinite(rate) & np.isfinite(thr)

        out[pid] = {
            "schedule": sched,
            "refits": info["refits"],
            "weeks": len(sched),
            "threshold": {
                "first": float(thr[0]), "last": float(thr[-1]),
                "median": float(np.median(thr)),
                "min": float(thr.min()), "max": float(thr.max()),
                # Spread relative to its own level: a threshold that lives at 4%
                # and one at 16% cannot be compared on absolute movement.
                "cv": float(thr.std() / thr.mean()) if thr.mean() else float("nan"),
                "median_weekly_step": float(np.median(steps)) if len(steps) else 0.0,
            },
            "trailing_hypo_rate": {
                "first": float(rate[0]), "last": float(rate[-1]),
                "min": float(np.nanmin(rate)), "max": float(np.nanmax(rate)),
            },
            "corr_threshold_vs_rate": (
                float(np.corrcoef(thr[finite], rate[finite])[0, 1])
                if finite.sum() > 3 else float("nan")
            ),
        }
    return out


def main() -> None:
    name = json.loads((ARTIFACTS_DIR / "selection.json").read_text())["selected"]
    fc = Forecaster(name)
    windows = build_windows(verbose=False)

    results = {}
    for cohort, ws in {"test": windows["test"],
                       "external": external_windows()}.items():
        results[cohort] = trajectories(ws, fc)
        print(f"\n=== {cohort}, {len(results[cohort])} wearers with a rolling history ===")
        print(f"{'wearer':<16}{'weeks':>6}{'refits':>8}{'first':>8}{'last':>8}"
              f"{'min':>7}{'max':>7}{'CV':>7}{'weekly step':>13}{'corr w/ TBR':>13}")
        for pid, r in results[cohort].items():
            th, tr = r["threshold"], r["corr_threshold_vs_rate"]
            print(f"{pid:<16}{r['weeks']:>6}{r['refits']:>8}{th['first']:>7.1%}"
                  f"{th['last']:>8.1%}{th['min']:>7.1%}{th['max']:>7.1%}"
                  f"{th['cv']:>7.2f}{th['median_weekly_step']:>12.1%}{tr:>13.2f}")

    with open(ARTIFACTS_DIR / "trajectory.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nWrote {ARTIFACTS_DIR / 'trajectory.json'}")


if __name__ == "__main__":
    main()
