"""Load every artefact the experiments produced, for the app to render.

Each loader returns None when its file is missing rather than raising, so the
app degrades to showing whatever has actually been run. A page that silently
omits a section is better than one that crashes because an experiment is still
in flight.
"""
from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from src.config import ARTIFACTS_DIR


@lru_cache(maxsize=32)
def load(name: str) -> dict | None:
    path = ARTIFACTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def model_table() -> pd.DataFrame | None:
    """Every model that was trained, with its held-out numbers."""
    sweep = load("sweep")
    if not sweep:
        return None
    rows = []
    for r in sweep["results"]:
        t = r["test"]
        rows.append({
            "Model": r["name"],
            "Params": r.get("n_params", 0),
            "RMSE": round(t["rmse"], 2),
            "MAE": round(t["mae"], 2),
            "RMSE (lows)": round(t["rmse_hypo"], 2),
            "Clarke A+B": round(t["clarke_ab"], 1),
        })
    for stem in ("tcn_prob", "tcn_cls", "tcn_mt"):
        blob = load(f"{stem}.metrics") or _metrics_file(stem)
        if blob and not any(r["Model"] == blob["name"] for r in rows):
            t = blob["test"]
            rows.append({
                "Model": blob["name"], "Params": blob.get("n_params", 0),
                "RMSE": round(t["rmse"], 2), "MAE": round(t["mae"], 2),
                "RMSE (lows)": round(t["rmse_hypo"], 2),
                "Clarke A+B": round(t["clarke_ab"], 1),
            })
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def _metrics_file(stem: str) -> dict | None:
    path = ARTIFACTS_DIR / f"{stem}.metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


def alarm_curves() -> dict | None:
    return load("alarm")


def matched_table() -> pd.DataFrame | None:
    """Recall at matched achieved false-alarm rates — the fair comparison."""
    m = load("matched")
    if not m:
        return None
    rates = m["rates"]
    rows = []
    for name, by_rate in m["test"].items():
        row = {"Model": name}
        for r in rates:
            row[f"{r:g} FA/day"] = f"{by_rate[str(r)] * 100:.1f}%"
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(df.columns[-1], ascending=False).reset_index(drop=True)


def calibration_table() -> pd.DataFrame | None:
    cal = load("calibration")
    if not cal:
        return None
    rows = []
    for cohort, by_warmup in cal["results"].items():
        for warmup, r in by_warmup.items():
            for label, key in [("shared", "population"), ("per-wearer", "calibrated")]:
                s = r[key]["per_patient_fa"]
                rows.append({
                    "Cohort": cohort, "Warm-up": warmup, "Threshold": label,
                    "Recall": f"{r[key]['recall']:.1%}",
                    "FA/day (median)": round(s["median"], 1),
                    "Spread across wearers": f"{s['min']:.1f} – {s['max']:.1f}",
                    "Within 2x of target": f"{s['within_2x_of_target']:.0%}",
                })
    return pd.DataFrame(rows)


def threshold_table() -> pd.DataFrame | None:
    """Each wearer's own alarm cutoff, next to how often they actually go low."""
    thr = load("thresholds")
    if not thr:
        return None
    rows = []
    for cohort, patients in thr.items():
        for r in patients:
            rows.append({
                "Cohort": cohort,
                "Wearer": r["pid"],
                "Threshold": None if r["thr"] is None else round(r["thr"] * 100, 1),
                "Time below 70": round(r["base"] * 100, 2),
                "Multiple of base rate": (None if r["thr"] is None
                                          else round(r["thr"] / r["base"], 1)),
                "Episodes warned": (None if r.get("recall") is None
                                    else round(r["recall"] * 100, 1)),
                "FA/day": None if r.get("fa") is None else round(r["fa"], 1),
                "Warm-up lows": r["lows_wu"],
            })
    return pd.DataFrame(rows)


def trajectory() -> dict | None:
    return load("trajectory")


def over_time_table() -> pd.DataFrame | None:
    ot = load("over_time")
    if not ot:
        return None
    rows = []
    for cohort, buckets in ot.items():
        for label, b in buckets.items():
            rows.append({
                "Cohort": cohort, "Since calibration": label,
                "Wearers": b["wearers"], "RMSE": round(b["rmse"], 2),
                "Time below 70": f"{b['hypo_rate']:.2%}",
                "Rolling recall": f"{b['rolling']['recall']:.1%}",
                "Fixed recall": f"{b['fixed']['recall']:.1%}",
                "Rolling FA/day": round(b["rolling"]["fa"], 1),
            })
    return pd.DataFrame(rows)


def personalized_table() -> pd.DataFrame | None:
    p = load("personalized")
    if not p:
        return None
    rows = []
    for cohort, blob in p.items():
        for variant, s in blob.get("summary", {}).items():
            rows.append({
                "Cohort": cohort,
                "Variant": variant.replace("_", " "),
                "Wearers": s["wearers"],
                "RMSE": round(s["rmse_mean"], 2),
                "vs shared": round(s["rmse_delta_vs_shared"], 2),
                "Wearers improved": f"{s['improved']:.0%}",
                "Episode recall": f"{s['episode_recall']:.1%}",
                "FA/day": round(s["false_alarms_per_day"], 1),
            })
    return pd.DataFrame(rows)


def multimodal_table() -> pd.DataFrame | None:
    mm = load("multimodal")
    if not mm:
        return None
    base = next((r["test"]["rmse"] for r in mm["results"]
                 if r["name"] == "cgm"), None)
    rows = []
    for r in mm["results"]:
        t = r["test"]
        rows.append({
            "Inputs": r["name"],
            "Channels": r["channels"][-1] if len(r["channels"]) > 1 else 1,
            "Params": r["n_params"],
            "val RMSE": round(r["val"]["rmse"], 2),
            "test RMSE": round(t["rmse"], 2),
            "vs CGM only": None if base is None else round(t["rmse"] - base, 2),
            "RMSE (lows)": round(t["rmse_hypo"], 2),
            "Clarke A+B": round(t["clarke_ab"], 1),
        })
    return pd.DataFrame(rows)


def external_summary() -> dict | None:
    return load("external")


def drift_table() -> pd.DataFrame | None:
    d = load("drift")
    if not d:
        return None
    rows = []
    for cohort, res in d["results"].items():
        for label, b in res["buckets"].items():
            rows.append({
                "Cohort": cohort, "Since calibration": label,
                "Wearers": b["patients"], "Wearer-days": round(b["days"]),
                "Time below 70": f"{b['hypo_rate']:.2%}",
                "FA/day": round(b["false_alarms_per_day"], 1),
                "Recall": f"{b['recall']:.1%}",
            })
    return pd.DataFrame(rows)


def within_patient_table() -> pd.DataFrame | None:
    w = load("within_patient")
    if not w:
        return None
    rows = []
    for cohort, blob in w.items():
        for label, r in blob["centred"].items():
            rows.append({
                "Cohort": cohort, "Since calibration": label,
                "Wearers": r["wearers"],
                "RMSE vs own average": round(r["mean_delta"], 2),
                "Worse than own average": f"{r['share_worse']:.0%}",
            })
    return pd.DataFrame(rows)
