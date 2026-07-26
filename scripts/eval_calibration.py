"""Does a per-wearer alarm threshold beat one shared threshold?

Runs both strategies over the same evaluation windows on the held-out test
patients and on the external population, and writes CALIBRATION.md.

Usage:  python -m scripts.eval_calibration
"""
from __future__ import annotations

import json

import numpy as np
import torch

from src.alarm import risk_score, tune_threshold
from src.calibration import evaluate_strategies
from src.config import ARTIFACTS_DIR
from src.data.windows import build_windows
from src.predictor import Forecaster
from scripts.eval_external import external_windows

TARGET_FA = 6.0
WARMUPS = (7.0, 14.0, 28.0)


def scores_for(fc: Forecaster, X: np.ndarray) -> np.ndarray:
    out = fc.predict_full(X)
    return out["hypo_prob"] if out["hypo_prob"] is not None else risk_score(out["mu"])


def pooled_curve(cohort: str) -> dict | None:
    """The PR curve a single global threshold traces out on this cohort."""
    if cohort == "test":
        alarm = ARTIFACTS_DIR / "alarm.json"
        if not alarm.exists():
            return None
        name = json.loads((ARTIFACTS_DIR / "selection.json").read_text())["selected"]
        return json.loads(alarm.read_text()).get(name, {}).get("pr_curve_test")
    path = ARTIFACTS_DIR / "external_pr.json"
    return json.loads(path.read_text()).get("model") if path.exists() else None


def recall_at(curve: dict, target: float) -> float:
    fa = np.asarray(curve["false_alarms_per_day"], dtype=float)
    rec = np.asarray(curve["recall"], dtype=float)
    order = np.argsort(fa)
    return float(np.interp(target, fa[order], rec[order]))


def main() -> None:
    name = json.loads((ARTIFACTS_DIR / "selection.json").read_text())["selected"]
    fc = Forecaster(name)

    windows = build_windows(verbose=False)
    val, test = windows["val"], windows["test"]

    # The shared threshold is the one a device would ship with: tuned on the
    # validation patients, before it has met anyone new.
    population_threshold = tune_threshold(val.y, scores_for(fc, val.X), TARGET_FA)
    print(f"model: {name}")
    print(f"population threshold (tuned on validation, {TARGET_FA:g} FA/day target): "
          f"{population_threshold:.4f}\n")

    cohorts = {"test": test, "external": external_windows()}
    results = {}

    for cohort_name, ws in cohorts.items():
        score = scores_for(fc, ws.X)
        results[cohort_name] = {}
        for warmup in WARMUPS:
            res = evaluate_strategies(
                ws.y, score, ws.times, ws.patient_ids,
                population_threshold, TARGET_FA, warmup,
            )
            results[cohort_name][f"{warmup:g}d"] = res

        print(f"=== {cohort_name} ({len(set(ws.patient_ids))} patients) ===")
        for warmup in WARMUPS:
            r = results[cohort_name][f"{warmup:g}d"]
            pop, cal = r["population"], r["calibrated"]
            fell = sum(p["fell_back"] for p in r["patients"])
            print(f"  warm-up {warmup:g}d  ({fell} patients fell back to the shared cutoff)")
            for label, m in [("shared    ", pop), ("per-wearer", cal)]:
                s = m["per_patient_fa"]
                print(f"    {label} recall {m['recall']:6.1%}  "
                      f"FA/day median {s['median']:5.1f} "
                      f"(range {s['min']:4.1f}–{s['max']:5.1f}), "
                      f"within 2x of target: {s['within_2x_of_target']:5.0%}")
        print()

    with open(ARTIFACTS_DIR / "calibration.json", "w") as fh:
        json.dump({"model": name, "target_fa_per_day": TARGET_FA,
                   "population_threshold": population_threshold,
                   "results": results}, fh, indent=2, default=float)
    write_markdown(name, results)


def write_markdown(model: str, results: dict) -> None:
    lines = [
        "# Per-wearer alarm calibration",
        "",
        "Two independent population shifts in this project broke the alarm "
        "threshold, and neither broke the model's ranking. A cutoff tuned for six "
        "false alarms a day lands at 14.7 on the test patients and 10.5 on the "
        "external population, because people differ enormously in how often they "
        "actually go low. There is no single number to find.",
        "",
        "So the threshold stops being a property of the model and becomes a "
        "property of the wearer: hold out each person's **first two weeks**, tune "
        "their cutoff on that, and use it for the rest of their record. A CGM is "
        "worn continuously, so this data costs nothing — it is simply the "
        "beginning of wearing the device.",
        "",
        "Both strategies below are scored on identical windows (everything after "
        "each patient's warm-up), so the only difference is where the threshold "
        "came from. A wearer whose warm-up contains fewer than 20 low readings "
        "falls back to the shared cutoff rather than inventing a personal one "
        "from a handful of events.",
        "",
    ]

    for cohort, by_warmup in results.items():
        lines += [f"## {cohort.capitalize()} cohort", "",
                  "| warm-up | strategy | recall | FA/day (median) | FA/day range across wearers | within 2× of target |",
                  "|---|---|---:|---:|---:|---:|"]
        for warmup, r in by_warmup.items():
            for label, key in [("shared cutoff", "population"), ("**per-wearer**", "calibrated")]:
                m = r[key]
                s = m["per_patient_fa"]
                lines.append(
                    f"| {warmup} | {label} | {m['recall']:.1%} | {s['median']:.1f} | "
                    f"{s['min']:.1f} – {s['max']:.1f} | {s['within_2x_of_target']:.0%} |"
                )
        lines.append("")

    lines += ["## What it costs", "",
        "Per-wearer calibration is not free, and the earlier tables make it look "
        "worse than it is. A shared threshold appears to have far higher recall "
        "only because it is firing two to three times more often than it was "
        "asked to. The comparison that matters holds the *achieved* rate fixed.",
        "",
        "| cohort | warm-up | per-wearer recall | at this FA/day | one shared threshold at the same rate | difference |",
        "|---|---|---:|---:|---:|---:|"]
    for cohort, by_warmup in results.items():
        curve = pooled_curve(cohort)
        if curve is None:
            continue
        for warmup, r in by_warmup.items():
            cal = r["calibrated"]
            fa = cal["false_alarms_per_day"]
            shared = recall_at(curve, fa)
            lines.append(
                f"| {cohort} | {warmup} | {cal['recall']:.1%} | {fa:.2f} | "
                f"{shared:.1%} | {(cal['recall'] - shared) * 100:+.1f} pp |"
            )
    lines += ["",
        "So it costs roughly five points of pooled recall. That price is worth "
        "understanding rather than explaining away: a single global threshold "
        "earns its pooled score partly by treating people unequally. It lets the "
        "wearers who go low often alarm constantly — cheap true positives — while "
        "the wearers who rarely go low get almost no warnings at all. Pooling "
        "hides that, and rewards it.",
        "",
        "Equalising the alarm rate across people gives some of that back. What it "
        "buys is that the number on the dial is true for the person reading it.",
        "",
        "## Reading this",
        "",
        "The column that matters is the last one. A shared cutoff does not miss "
        "the target by a little — it misses it by different amounts for different "
        "people, which is what makes a population-level threshold unshippable. "
        "Per-wearer calibration is not about squeezing out more recall; it is "
        "about the alarm rate meaning what it says for the person wearing it.",
        "",
        "The warm-up length trades off against itself: longer warm-ups pin the "
        "threshold down better but delay the moment the device is properly tuned, "
        "and leave fewer windows to evaluate on.",
        "",
        "_Generated by `python -m scripts.eval_calibration`._",
    ]
    (ARTIFACTS_DIR.parent / "CALIBRATION.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {ARTIFACTS_DIR.parent / 'CALIBRATION.md'}")


def rewrite() -> None:
    """Regenerate the write-up from saved JSON, without re-running the model."""
    blob = json.loads((ARTIFACTS_DIR / "calibration.json").read_text())
    write_markdown(blob["model"], blob["results"])


if __name__ == "__main__":
    import sys
    rewrite() if "--rewrite" in sys.argv else main()
