"""Apply the shipped model, unchanged, to the external population.

No retraining, no refitting, no re-tuning. The alarm threshold is the one
already tuned on the original validation patients. If the numbers hold up here
they mean something; if they collapse, that is the more useful result.

Usage:  python -m scripts.eval_external
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from src.alarm import alarm_metrics, risk_score
from src.config import ARTIFACTS_DIR, CACHE_DIR, HYPO_THRESHOLD
from src.data.windows import WindowSet, _windows_for_patient
from src.config import HISTORY_STEPS
from src.metrics import HEADER, evaluate, format_row
from src.models.baselines import persistence
from src.predictor import Forecaster, tuned_threshold
from scripts.alarm_report import BUDGETS


def external_windows() -> WindowSet:
    path = CACHE_DIR / "cgm_external.parquet"
    df = pd.read_parquet(path)
    parts = []
    for pid, grp in df.groupby("patient_id", sort=True):
        grp = grp.sort_values("datetime")
        built = _windows_for_patient(pid, grp["glucose"], grp["datetime"], HISTORY_STEPS)
        if built is not None:
            parts.append(built)
    return WindowSet(
        X=np.concatenate([p[0] for p in parts]),
        y=np.concatenate([p[1] for p in parts]),
        patient_ids=np.concatenate([p[2] for p in parts]),
        times=np.concatenate([p[3] for p in parts]),
    )


def main() -> None:
    selection = ARTIFACTS_DIR / "selection.json"
    name = json.loads(selection.read_text())["selected"] if selection.exists() else "tcn_prob"

    ext = external_windows()
    n_patients = len(set(ext.patient_ids))
    hypo_rate = float((ext.y < HYPO_THRESHOLD).mean())
    print(f"external set: {n_patients} patients, {len(ext):,} windows, "
          f"hypo rate {hypo_rate:.2%}\n")

    fc = Forecaster(name)
    out = fc.predict_full(ext.X)
    mu = out["mu"]

    m_model = evaluate(ext.y, mu)
    m_persist = evaluate(ext.y, persistence(ext.X))
    print(HEADER)
    print(format_row("persistence", m_persist))
    print(format_row(f"{name}", m_model))

    score = (out["hypo_prob"] if out["hypo_prob"] is not None
             else risk_score(mu))
    rows = {}
    print(f"\nAlarm, thresholds carried over unchanged from the original validation split:")
    for b in BUDGETS:
        thr = tuned_threshold(fc, f"{b:g}/day")
        m = alarm_metrics(ext.y, score, thr)
        rows[f"{b:g}/day"] = m
        print(f"  tuned for ≤{b:g} FA/day -> recall {m['recall']:6.1%}  "
              f"precision {m['precision']:5.1%}  achieved {m['false_alarms_per_day']:5.1f} FA/day")

    p_score = risk_score(persistence(ext.X))
    print("\n  persistence at the same achieved false-alarm rates, for reference:")
    for b in BUDGETS:
        target = rows[f"{b:g}/day"]["false_alarms_per_day"]
        thr = _match_fa(ext.y, p_score, target)
        m = alarm_metrics(ext.y, p_score, thr)
        print(f"    ~{target:4.1f} FA/day -> recall {m['recall']:6.1%}")

    payload = {
        "model": name,
        "patients": n_patients,
        "windows": int(len(ext)),
        "hypo_rate": hypo_rate,
        "regression": {"model": m_model, "persistence": m_persist},
        "alarm": rows,
    }
    with open(ARTIFACTS_DIR / "external.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {ARTIFACTS_DIR / 'external.json'}")


def _match_fa(y: np.ndarray, score: np.ndarray, target_per_day: float) -> float:
    """Threshold that yields roughly `target_per_day` false alarms."""
    from src.alarm import tune_threshold
    return tune_threshold(y, score, target_per_day)


if __name__ == "__main__":
    main()
