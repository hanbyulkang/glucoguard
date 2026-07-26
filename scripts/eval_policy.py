"""Three alarm policies, scored the way a wearer would count them.

* **fixed**, one threshold fitted on the wearer's first two weeks, never touched
  again.
* **rolling**, re-fitted every week on the trailing four weeks, so it follows
  the person.
* **shared**, one population threshold for everybody, as a floor.

All three are scored on episodes and alarm events rather than on individual
five-minute readings, with a refractory period so a burst counts once.

Usage:  python -m scripts.eval_policy
"""
from __future__ import annotations

import json

import numpy as np

from src.alarm_policy import (
    REFRACTORY_MINUTES,
    event_metrics,
    rolling_thresholds,
    tune_event_threshold,
)
from src.calibration import split_by_time
from src.config import ARTIFACTS_DIR, HYPO_THRESHOLD
from src.data.windows import build_windows
from src.predictor import Forecaster
from scripts.eval_calibration import scores_for
from scripts.eval_external import external_windows

TARGET = 6.0            # false alarm *events* per day
WARMUP_DAYS = 14.0
MIN_WARMUP_LOWS = 20


def pooled(records: list) -> dict:
    """Aggregate per-wearer event counts into one honest total."""
    eps = sum(r.episodes for r in records)
    warned = sum(r.episodes_warned for r in records)
    days = sum(r.days for r in records)
    fa = sum(r.false_alarm_events for r in records)
    leads = [r.median_lead_minutes for r in records
             if np.isfinite(r.median_lead_minutes)]
    per_wearer_fa = np.array([r.false_alarms_per_day for r in records])
    return {
        "wearers": len(records),
        "episodes": eps,
        "episode_recall": warned / eps if eps else float("nan"),
        "false_alarms_per_day": fa / days if days else float("nan"),
        "alarms_per_day": sum(r.alarm_events for r in records) / days if days else float("nan"),
        "median_lead_minutes": float(np.median(leads)) if leads else float("nan"),
        "per_wearer_fa": {
            "median": float(np.median(per_wearer_fa)),
            "min": float(per_wearer_fa.min()),
            "max": float(per_wearer_fa.max()),
            "within_2x": float(np.mean((per_wearer_fa >= TARGET / 2)
                                       & (per_wearer_fa <= TARGET * 2))),
        },
    }


def run_cohort(ws, fc: Forecaster, shared_threshold: float) -> dict:
    score = scores_for(fc, ws.X)
    by_policy: dict[str, list] = {"shared": [], "fixed": [], "rolling": []}
    rolling_info = []

    for pid in sorted(set(ws.patient_ids)):
        sel = ws.patient_ids == pid
        y, s = ws.y[sel], score[sel]
        t = np.asarray(ws.times[sel], dtype="datetime64[ns]")

        warmup = split_by_time(t, WARMUP_DAYS)
        if int((y[warmup] < HYPO_THRESHOLD).sum()) < MIN_WARMUP_LOWS:
            continue
        evaluation = ~warmup
        if evaluation.sum() == 0:
            continue

        y_e, s_e, t_e = y[evaluation], s[evaluation], t[evaluation]

        by_policy["shared"].append(event_metrics(y_e, s_e >= shared_threshold))

        fixed = tune_event_threshold(y[warmup], s[warmup], TARGET)
        by_policy["fixed"].append(event_metrics(y_e, s_e >= fixed))

        thr, info = rolling_thresholds(y, s, t, TARGET)
        thr_e = thr[evaluation]
        # Before the first refit there is no rolling threshold; fall back to the
        # fixed one so the comparison covers identical windows.
        thr_e = np.where(np.isfinite(thr_e), thr_e, fixed)
        by_policy["rolling"].append(event_metrics(y_e, s_e >= thr_e))
        rolling_info.append(info)

    out = {k: pooled(v) for k, v in by_policy.items() if v}
    out["rolling"]["refits_median"] = float(
        np.median([i["refits"] for i in rolling_info])) if rolling_info else 0.0
    return out


def main() -> None:
    name = json.loads((ARTIFACTS_DIR / "selection.json").read_text())["selected"]
    fc = Forecaster(name)
    windows = build_windows(verbose=False)
    val = windows["val"]

    shared = tune_event_threshold(val.y, scores_for(fc, val.X), TARGET)
    print(f"model {name}; shared threshold from validation: {shared:.4f}")
    print(f"target {TARGET:g} false alarm EVENTS per day, "
          f"{REFRACTORY_MINUTES}-minute refractory period\n")

    results = {}
    for cohort, ws in {"test": windows["test"], "external": external_windows()}.items():
        results[cohort] = run_cohort(ws, fc, shared)
        print(f"=== {cohort} ===")
        print(f"{'policy':<10s}{'wearers':>8s}{'episodes':>10s}{'recall':>9s}"
              f"{'FA/day':>9s}{'alarms/day':>12s}{'lead':>8s}{'within2x':>10s}")
        for policy in ("shared", "fixed", "rolling"):
            m = results[cohort].get(policy)
            if not m:
                continue
            print(f"{policy:<10s}{m['wearers']:>8d}{m['episodes']:>10,}"
                  f"{m['episode_recall']:>9.1%}{m['false_alarms_per_day']:>9.1f}"
                  f"{m['alarms_per_day']:>12.1f}{m['median_lead_minutes']:>7.0f}m"
                  f"{m['per_wearer_fa']['within_2x']:>10.0%}")
        print()

    with open(ARTIFACTS_DIR / "policy.json", "w") as fh:
        json.dump({"model": name, "target": TARGET,
                   "refractory_minutes": REFRACTORY_MINUTES,
                   "shared_threshold": shared, "results": results},
                  fh, indent=2, default=float)
    write_markdown(results)


def write_markdown(results: dict) -> None:
    lines = [
        "# Counting alarms the way a wearer counts them",
        "",
        "Every number before this one treated each five-minute reading as its own "
        "alarm opportunity. A device does not work that way and neither does a "
        "person. Under per-reading accounting, half an hour of nuisance alarming "
        "is six false alarms, and one low the model catches slightly late is "
        "several misses and several hits at the same time.",
        "",
        f"So: an alarm fires and then stays quiet for {REFRACTORY_MINUTES} minutes. "
        "A **low episode** counts as warned if the device made a sound in the hour "
        "before glucose crossed 70. An **alarm event** counts as false only if no "
        "low followed it. Alarms during an ongoing low are not false, the wearer "
        "is low and the device is right to be noisy.",
        "",
        "Three policies, all scored on the same windows (everything after each "
        "wearer's two-week warm-up):",
        "",
        "- **shared**, one threshold for everybody, fitted on the validation wearers",
        "- **fixed**, fitted once on this wearer's first fortnight, then never touched",
        "- **rolling**, re-fitted weekly on this wearer's trailing four weeks",
        "",
    ]
    for cohort, res in results.items():
        lines += [
            f"## {cohort.capitalize()} cohort",
            "",
            "| policy | wearers | low episodes | **episodes warned** | false alarms/day | total alarms/day | median warning | wearers within 2× of target |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for policy in ("shared", "fixed", "rolling"):
            m = res.get(policy)
            if not m:
                continue
            label = f"**{policy}**" if policy == "rolling" else policy
            lines.append(
                f"| {label} | {m['wearers']} | {m['episodes']:,} | "
                f"**{m['episode_recall']:.1%}** | {m['false_alarms_per_day']:.1f} | "
                f"{m['alarms_per_day']:.1f} | {m['median_lead_minutes']:.0f} min | "
                f"{m['per_wearer_fa']['within_2x']:.0%} |"
            )
        lines.append("")

    lines += [
        "## Why this is not a metric trick",
        "",
        "It would be, if the change only ever moved numbers upward. It does not: "
        "the same de-duplication that raises recall also strips out most of what "
        "used to be counted as false alarms, so the threshold has to be re-tuned "
        "in event units to hit the same budget. What changes is that both sides of "
        "the trade are now expressed in what the wearer experiences, how many "
        "times it interrupted them, and how many of their lows it saw coming.",
        "",
        "The per-reading numbers elsewhere in this repository are not wrong, they "
        "answer a different and less useful question: given a randomly chosen "
        "five-minute window that happens to be low, was the model's output under "
        "70 at that instant.",
        "",
        "_Generated by `python -m scripts.eval_policy`._",
    ]
    (ARTIFACTS_DIR.parent / "ALARM_POLICY.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {ARTIFACTS_DIR.parent / 'ALARM_POLICY.md'}")


if __name__ == "__main__":
    main()
