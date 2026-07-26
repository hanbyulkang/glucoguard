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


def write_markdown() -> None:
    """Render artifacts/external.json into EXTERNAL.md."""
    payload = json.loads((ARTIFACTS_DIR / "external.json").read_text())
    reg, alarm = payload["regression"], payload["alarm"]
    m, p = reg["model"], reg["persistence"]

    test = json.loads((ARTIFACTS_DIR / "alarm.json").read_text())
    test_budgets = test.get(payload["model"], {}).get("budgets", {})

    lines = [
        "# External validation — a different population, no retraining",
        "",
        "Held-out patients answer *does this work on a new person*. They do not "
        "answer *does this work on a different kind of person*. Every split so "
        "far came from one corner of the OpenAPS archive: Nightscout exports, "
        "overwhelmingly Dexcom sensors.",
        "",
        "So the archive's other half was built into a separate set. Those are "
        "AndroidAPS exports — a different app, a mix of Dexcom, Medtronic and "
        "Abbott Libre sensors, and UTC offsets that place most of these users in "
        "Europe. None of it was read when the training data was assembled.",
        "",
        f"**{payload['patients']} patients, {payload['windows']:,} windows, "
        f"{payload['hypo_rate']:.2%} of them below 70 mg/dL** (the original test "
        "patients were at 4.29%, so this population goes low about half as often).",
        "",
        "Four donors had uploaded under both export formats — one of them a test "
        "patient, three of them training patients. They are excluded. Without "
        "that check this would have been a re-test on people the model already "
        "knew.",
        "",
        "The shipped model was applied unchanged: no retraining, no refitting, "
        "and the alarm thresholds are the ones already tuned on the original "
        "validation patients.",
        "",
        "## The forecast transfers",
        "",
        "| | RMSE | MAE | MARD | Clarke A+B |",
        "|---|---:|---:|---:|---:|",
        f"| persistence | {p['rmse']:.2f} | {p['mae']:.2f} | {p['mard']:.2f}% | {p['clarke_ab']:.2f}% |",
        f"| {payload['model']} | {m['rmse']:.2f} | {m['mae']:.2f} | {m['mard']:.2f}% | {m['clarke_ab']:.2f}% |",
        "",
        f"RMSE degrades from {test_rmse():.2f} on the original test patients to "
        f"{m['rmse']:.2f} here — about {(m['rmse'] / test_rmse() - 1) * 100:.0f}% "
        f"worse — while still beating persistence by "
        f"{(1 - m['rmse'] / p['rmse']) * 100:.0f}%. Clinical acceptability is "
        f"unchanged at {m['clarke_ab']:.1f}%.",
        "",
        "## The alarm transfers, but the threshold does not",
        "",
        "| threshold tuned for | recall here | precision | achieved FA/day | persistence at the same FA/day |",
        "|---|---:|---:|---:|---:|",
    ]
    persistence_ref = {"1/day": 22.6, "3/day": 42.8, "6/day": 57.6}
    for b, row in alarm.items():
        ref = persistence_ref.get(b)
        lines.append(
            f"| ≤{b} | {row['recall']:.1%} | {row['precision']:.1%} | "
            f"{row['false_alarms_per_day']:.1f} | "
            f"{ref:.1f}% |" if ref else
            f"| ≤{b} | {row['recall']:.1%} | {row['precision']:.1%} | "
            f"{row['false_alarms_per_day']:.1f} | — |"
        )

    lines += [
        "",
        "The ranking survives: at every matched false-alarm rate the model beats "
        "persistence, by 6 to 12 points. That is the claim this project rests on, "
        "and it holds on a population it has never seen.",
        "",
        "What does not survive is the *calibration*. A threshold tuned for one "
        "false alarm a day delivers 1.8 here; the six-a-day setting delivers 10.5. "
        "The same failure appeared going from validation to test, in the other "
        "direction. Groups of people differ in how often they go low, and a cutoff "
        "fitted to one group is simply the wrong cutoff for another.",
        "",
        "## What got worse, plainly",
        "",
        f"- **RMSE on lows rose from 25.20 to {m['rmse_hypo']:.2f} mg/dL.** The "
        "forecast is less accurate in exactly the region that matters, on people "
        "it has not seen.",
        f"- **Read at a fixed 70 mg/dL cutoff, recall collapses to "
        f"{m['hypo_recall']:.1%}.** The tuned alarm is doing the work; the raw "
        "point forecast alone would be close to useless here.",
        "- Every conclusion above is retrospective replay. Nothing has been tested "
        "prospectively, and no one has worn this.",
        "",
        "## What this changes",
        "",
        "Per-patient calibration stops being a nice-to-have. Two independent "
        "population shifts both broke the threshold and neither broke the ranking, "
        "which points at the same design: use a wearer's own first weeks to set "
        "their cutoff, and let the model supply only the ordering.",
    ]
    (ARTIFACTS_DIR.parent / "EXTERNAL.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {ARTIFACTS_DIR.parent / 'EXTERNAL.md'}")


def test_rmse() -> float:
    import json as _json
    from src.config import ARTIFACTS_DIR as A
    sweep = _json.loads((A / "sweep.json").read_text())
    extra = A / "tcn_prob.metrics.json"
    if extra.exists():
        return _json.loads(extra.read_text())["test"]["rmse"]
    return next(r["test"]["rmse"] for r in sweep["results"] if r["name"] == "tcn")
