"""Does anything actually improve as the weeks accumulate?

Worth being precise about what could. The network is frozen after training — it
never sees another gradient — so its forecast accuracy has no mechanism to
improve with wear, and if RMSE moves over time that is the wearer changing, not
the model learning.

What *can* improve is the alarm. The threshold re-fits weekly, so the question
is whether it converges on something that serves the wearer better as it
accumulates history, or whether the first fit was already as good as it gets.

Thresholds are read back from `trajectory.json` rather than re-tuned, so each
reading is scored against the cutoff that was actually in force that week.

Usage:  python -m scripts.eval_over_time
"""
from __future__ import annotations

import json

import numpy as np

from src.alarm_policy import event_metrics, tune_event_threshold
from src.calibration import split_by_time
from src.config import ARTIFACTS_DIR, HYPO_THRESHOLD
from src.data.windows import build_windows
from src.metrics import evaluate
from src.predictor import Forecaster
from scripts.eval_calibration import scores_for
from scripts.eval_external import external_windows

TARGET = 6.0
WARMUP_DAYS = 14.0
REFIT_DAYS = 7.0

BUCKETS = [(0, 30), (30, 90), (90, 180), (180, 365), (365, 730), (730, 10_000)]
LABELS = ["month 1", "months 1–3", "months 3–6", "months 6–12",
          "year 1–2", "year 2+"]


def thresholds_from_schedule(schedule: list[dict], times: np.ndarray,
                             start: np.datetime64) -> np.ndarray:
    """Rebuild the per-reading cutoff from the saved weekly refit schedule."""
    out = np.full(len(times), np.nan)
    age = (times - start) / np.timedelta64(1, "D")
    for entry in schedule:
        lo = entry["days_since_start"]
        applies = (age >= lo) & (age < lo + REFIT_DAYS)
        out[applies] = entry["threshold"]
    return out


def analyse(ws, fc: Forecaster, traj: dict) -> dict:
    score = scores_for(fc, ws.X)
    buckets = {label: {"y": [], "pred": [], "roll": [], "fix": []} for label in LABELS}

    for pid in sorted(set(ws.patient_ids)):
        if pid not in traj:
            continue
        sel = ws.patient_ids == pid
        y, s = ws.y[sel], score[sel]
        t = np.asarray(ws.times[sel], dtype="datetime64[ns]")
        order = np.argsort(t)
        y, s, t = y[order], s[order], t[order]
        mu = fc.predict(ws.X[sel][order])

        warm = split_by_time(t, WARMUP_DAYS)
        if int((y[warm] < HYPO_THRESHOLD).sum()) < 20:
            continue
        fixed = tune_event_threshold(y[warm], s[warm], TARGET)
        rolling = thresholds_from_schedule(traj[pid]["schedule"], t, t.min())
        rolling = np.where(np.isfinite(rolling), rolling, fixed)

        age = (t - t[warm].max()) / np.timedelta64(1, "D")
        for (lo, hi), label in zip(BUCKETS, LABELS):
            m = (~warm) & (age >= lo) & (age < hi)
            if m.sum() < 500:
                continue
            buckets[label]["y"].append(y[m])
            buckets[label]["pred"].append(mu[m])
            buckets[label]["roll"].append(s[m] >= rolling[m])
            buckets[label]["fix"].append(s[m] >= fixed)

    out = {}
    for label in LABELS:
        b = buckets[label]
        if not b["y"]:
            continue
        y = np.concatenate(b["y"])
        reg = evaluate(y, np.concatenate(b["pred"]))
        roll = event_metrics(y, np.concatenate(b["roll"]))
        fix = event_metrics(y, np.concatenate(b["fix"]))
        out[label] = {
            "wearers": len(b["y"]),
            "days": len(y) * 5 / (60 * 24),
            "hypo_rate": float((y < HYPO_THRESHOLD).mean()),
            "rmse": reg["rmse"],
            "rolling": {"recall": roll.episode_recall,
                        "fa": roll.false_alarms_per_day},
            "fixed": {"recall": fix.episode_recall,
                      "fa": fix.false_alarms_per_day},
        }
    return out


def main() -> None:
    name = json.loads((ARTIFACTS_DIR / "selection.json").read_text())["selected"]
    fc = Forecaster(name)
    traj = json.loads((ARTIFACTS_DIR / "trajectory.json").read_text())
    windows = build_windows(verbose=False)

    results = {}
    for cohort, ws in {"test": windows["test"],
                       "external": external_windows()}.items():
        results[cohort] = analyse(ws, fc, traj.get(cohort, {}))
        print(f"\n=== {cohort} ===")
        print(f"{'since calibration':<20}{'wearers':>8}{'RMSE':>8}{'TBR':>8}"
              f"{'  rolling: recall  FA':>24}{'  fixed: recall  FA':>22}")
        for label, b in results[cohort].items():
            print(f"{label:<20}{b['wearers']:>8}{b['rmse']:>8.2f}"
                  f"{b['hypo_rate']:>7.2%}"
                  f"{b['rolling']['recall']:>16.1%}{b['rolling']['fa']:>7.1f}"
                  f"{b['fixed']['recall']:>15.1%}{b['fixed']['fa']:>7.1f}")

    with open(ARTIFACTS_DIR / "over_time.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nWrote {ARTIFACTS_DIR / 'over_time.json'}")


if __name__ == "__main__":
    main()
