"""Does the forecast decay for a given person, or did the cohort just change?

Pooled RMSE rose from 17.1 to 21.0 across the time buckets, which looks like a
model going stale. It cannot be caused by recalibration — the threshold decides
when to alarm and never touches the prediction — so either the wearers are
drifting away from what the network learned, or the later buckets simply contain
different people. The wearer count falls from 7 to 4 and 25 to 9, so the second
explanation is available for free.

This controls for it by comparing each wearer against themselves: RMSE per time
bucket, centred on that wearer's own mean. If the forecast really decays, every
wearer's curve should slope up. If it was composition, the centred curves should
be flat and the pooled rise should vanish.

Usage:  python -m scripts.eval_within_patient
"""
from __future__ import annotations

import json

import numpy as np

from src.calibration import split_by_time
from src.config import ARTIFACTS_DIR
from src.data.windows import build_windows
from src.predictor import Forecaster
from scripts.eval_external import external_windows
from scripts.eval_over_time import BUCKETS, LABELS, WARMUP_DAYS

MIN_WINDOWS = 2000      # a bucket needs this much data to give a stable RMSE


def per_patient_rmse(ws, fc: Forecaster) -> dict:
    mu_all = fc.predict(ws.X)
    out = {}
    for pid in sorted(set(ws.patient_ids)):
        sel = ws.patient_ids == pid
        y, mu = ws.y[sel], mu_all[sel]
        t = np.asarray(ws.times[sel], dtype="datetime64[ns]")
        order = np.argsort(t)
        y, mu, t = y[order], mu[order], t[order]

        warm = split_by_time(t, WARMUP_DAYS)
        age = (t - t[warm].max()) / np.timedelta64(1, "D")

        buckets = {}
        for (lo, hi), label in zip(BUCKETS, LABELS):
            m = (~warm) & (age >= lo) & (age < hi)
            if m.sum() < MIN_WINDOWS:
                continue
            err = mu[m] - y[m]
            buckets[label] = {
                "rmse": float(np.sqrt(np.mean(err**2))),
                "windows": int(m.sum()),
                "glucose_sd": float(np.std(y[m])),
                "calendar_year": float(
                    t[m].astype("datetime64[Y]").astype(int).mean() + 1970
                ),
            }
        if len(buckets) >= 2:
            out[pid] = buckets
    return out


def summarise(per_patient: dict) -> dict:
    """Pooled-looking table, but with each wearer centred on their own mean."""
    rows = {label: [] for label in LABELS}
    for buckets in per_patient.values():
        vals = np.array([b["rmse"] for b in buckets.values()])
        centre = vals.mean()
        for label, b in buckets.items():
            rows[label].append(b["rmse"] - centre)

    out = {}
    for label, deltas in rows.items():
        if not deltas:
            continue
        d = np.array(deltas)
        out[label] = {
            "wearers": len(d),
            "mean_delta": float(d.mean()),
            "median_delta": float(np.median(d)),
            "share_worse": float((d > 0).mean()),
        }
    return out


def main() -> None:
    name = json.loads((ARTIFACTS_DIR / "selection.json").read_text())["selected"]
    fc = Forecaster(name)
    windows = build_windows(verbose=False)

    results = {}
    for cohort, ws in {"test": windows["test"],
                       "external": external_windows()}.items():
        per_patient = per_patient_rmse(ws, fc)
        centred = summarise(per_patient)
        results[cohort] = {"per_patient": per_patient, "centred": centred}

        print(f"\n=== {cohort} — RMSE per wearer, each centred on their own mean ===")
        print(f"{'since calibration':<20}{'wearers':>9}{'ΔRMSE':>9}{'worse than own avg':>21}")
        for label, r in centred.items():
            print(f"{label:<20}{r['wearers']:>9}{r['mean_delta']:>+9.2f}"
                  f"{r['share_worse']:>20.0%}")

        # A within-wearer slope: does each person's own RMSE trend upward?
        slopes = []
        for buckets in per_patient.values():
            if len(buckets) < 3:
                continue
            idx = [LABELS.index(l) for l in buckets]
            vals = [b["rmse"] for b in buckets.values()]
            slopes.append(np.polyfit(idx, vals, 1)[0])
        if slopes:
            s = np.array(slopes)
            print(f"\n  within-wearer slope: median {np.median(s):+.2f} mg/dL per bucket, "
                  f"{(s > 0).mean():.0%} of {len(s)} wearers trending worse")

    with open(ARTIFACTS_DIR / "within_patient.json", "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nWrote {ARTIFACTS_DIR / 'within_patient.json'}")


if __name__ == "__main__":
    main()
