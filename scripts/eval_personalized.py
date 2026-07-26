"""Is a model fine-tuned on one wearer better for that wearer?

Three variants, scored on the same windows:

* **shared**, the global model, frozen, as shipped.
* **personal (once)**, fine-tuned once on the wearer's first 90 days.
* **personal (refreshed)**, re-fine-tuned every 90 days on the trailing 90,
  always restarting from the shared weights.

Every personal model is fitted only on readings from strictly before the block
it predicts. Where a wearer has too little history to fine-tune on, the shared
model is used and that fact is reported rather than hidden.

Usage:  python -m scripts.eval_personalized [--cohort test]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from src.alarm import risk_score
from src.alarm_policy import event_metrics, tune_event_threshold
from src.calibration import split_by_time
from src.config import ARTIFACTS_DIR, HYPO_THRESHOLD
from src.data.windows import build_windows
from src.metrics import evaluate
from src.personalize import batched_predict, fine_tune, rolling_personal_predictions
from src.predictor import Forecaster
from src.train import pick_device
from scripts.eval_external import external_windows

TARGET = 6.0
WARMUP_DAYS = 90.0       # a personal model needs more history than a threshold


def to_score(pred: np.ndarray, fc: Forecaster) -> np.ndarray:
    if pred.ndim == 1:
        return risk_score(pred)
    if fc.classifies:
        return 1.0 / (1.0 + np.exp(-pred[:, -1]))
    if fc.probabilistic:
        return risk_score(pred[:, 0], pred[:, 1])
    return risk_score(pred[:, 0])


def score_variant(y, pred, fc, warm_y, warm_pred) -> dict:
    """RMSE plus an alarm tuned on this wearer's warm-up, in event units."""
    mu = pred[:, 0] if pred.ndim == 2 else pred
    reg = evaluate(y, mu)
    thr = tune_event_threshold(warm_y, to_score(warm_pred, fc), TARGET)
    m = event_metrics(y, to_score(pred, fc) >= thr)
    return {"rmse": reg["rmse"], "rmse_hypo": reg["rmse_hypo"],
            "episode_recall": m.episode_recall,
            "false_alarms_per_day": m.false_alarms_per_day}


def run(cohort: str, ws, fc: Forecaster, device) -> dict:
    base = fc.model
    out, skipped = {}, 0

    for pid in sorted(set(ws.patient_ids)):
        sel = ws.patient_ids == pid
        X, y = ws.X[sel], ws.y[sel]
        t = np.asarray(ws.times[sel], dtype="datetime64[ns]")
        order = np.argsort(t)
        X, y, t = X[order], y[order], t[order]

        warm = split_by_time(t, WARMUP_DAYS)
        evaluation = ~warm
        if evaluation.sum() < 5_000 or int((y[warm] < HYPO_THRESHOLD).sum()) < 40:
            skipped += 1
            continue

        shared_pred = batched_predict(base, X, device)
        variants = {"shared": shared_pred[evaluation]}

        personal, _ = fine_tune(base, X[warm], y[warm], device,
                                fc.probabilistic, fc.classifies)
        if personal is None:
            skipped += 1
            continue
        variants["personal_once"] = batched_predict(personal, X[evaluation], device)

        rolled, reports = rolling_personal_predictions(
            base, X, y, t, device, fc.probabilistic, fc.classifies)
        rolled_eval = rolled[evaluation]
        # Fall back to the shared model wherever no personal one was ready.
        missing = ~np.isfinite(rolled_eval).all(axis=-1) if rolled_eval.ndim == 2 \
            else ~np.isfinite(rolled_eval)
        rolled_eval = np.where(missing[..., None] if rolled_eval.ndim == 2 else missing,
                               variants["shared"], rolled_eval)
        variants["personal_rolling"] = rolled_eval

        row = {}
        for name, pred in variants.items():
            warm_pred = (shared_pred[warm] if name == "shared"
                         else batched_predict(personal, X[warm], device))
            row[name] = score_variant(y[evaluation], pred, fc, y[warm], warm_pred)
        row["_coverage"] = float(1 - missing.mean())
        row["_refreshes"] = len([r for r in reports if r.steps > 0])
        out[pid] = row
        print(f"  {pid}: shared RMSE {row['shared']['rmse']:.2f} -> "
              f"once {row['personal_once']['rmse']:.2f} -> "
              f"rolling {row['personal_rolling']['rmse']:.2f}  "
              f"(recall {row['shared']['episode_recall']:.0%} -> "
              f"{row['personal_rolling']['episode_recall']:.0%}, "
              f"{row['_refreshes']} refreshes)", flush=True)

    return {"per_patient": out, "skipped": skipped}


def summarise(per_patient: dict) -> dict:
    names = ["shared", "personal_once", "personal_rolling"]
    out = {}
    for name in names:
        rmse = np.array([r[name]["rmse"] for r in per_patient.values()])
        rec = np.array([r[name]["episode_recall"] for r in per_patient.values()])
        fa = np.array([r[name]["false_alarms_per_day"] for r in per_patient.values()])
        base = np.array([r["shared"]["rmse"] for r in per_patient.values()])
        out[name] = {
            "wearers": len(rmse),
            "rmse_mean": float(rmse.mean()),
            "rmse_delta_vs_shared": float((rmse - base).mean()),
            "improved": float((rmse < base).mean()),
            "episode_recall": float(rec.mean()),
            "false_alarms_per_day": float(fa.mean()),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", default="both", choices=["test", "external", "both"])
    args = p.parse_args()

    name = json.loads((ARTIFACTS_DIR / "selection.json").read_text())["selected"]
    fc = Forecaster(name)
    device = pick_device()
    windows = build_windows(verbose=False)

    cohorts = {}
    if args.cohort in ("test", "both"):
        cohorts["test"] = windows["test"]
    if args.cohort in ("external", "both"):
        cohorts["external"] = external_windows()

    results = {}
    for cohort, ws in cohorts.items():
        print(f"\n=== {cohort} ===", flush=True)
        res = run(cohort, ws, fc, device)
        res["summary"] = summarise(res["per_patient"])
        results[cohort] = res

        print(f"\n{'variant':<20}{'wearers':>8}{'RMSE':>8}{'vs shared':>11}"
              f"{'improved':>10}{'recall':>9}{'FA/day':>8}")
        for v, s in res["summary"].items():
            print(f"{v:<20}{s['wearers']:>8}{s['rmse_mean']:>8.2f}"
                  f"{s['rmse_delta_vs_shared']:>+11.2f}{s['improved']:>10.0%}"
                  f"{s['episode_recall']:>9.1%}{s['false_alarms_per_day']:>8.1f}")
        print(f"({res['skipped']} wearers skipped for lack of history)")

    with open(ARTIFACTS_DIR / "personalized.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nWrote {ARTIFACTS_DIR / 'personalized.json'}")


if __name__ == "__main__":
    main()
