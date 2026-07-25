"""Evaluation metrics.

Accuracy in mg/dL is not the whole story for a glucose forecaster. A model can
win on RMSE while being useless at the only moment that matters clinically —
the approach to hypoglycaemia — because lows are rare and the error there is
averaged away. So alongside the regression numbers we report how often the
model would actually have caught a low, and how often it would have cried wolf.
"""
from __future__ import annotations

import numpy as np

from src.config import HYPER_THRESHOLD, HYPO_THRESHOLD


def _clarke_zones(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Clarke Error Grid zone for each point, as 0..4 meaning A..E.

    Clarke et al., *Diabetes Care* 1987. Zone A is clinically accurate, B is
    benign error, and C/D/E are the ones that would lead to wrong treatment.
    """
    ref, pred = reference.astype(float), prediction.astype(float)
    zones = np.full(ref.shape, 1, dtype=np.int8)   # default to B

    # Zone A: within 20% of reference, or both under 70.
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(pred - ref) / np.where(ref == 0, np.nan, ref)
    zone_a = (rel <= 0.2) | ((pred < 70) & (ref < 70))

    # Zone E: treatment would be the opposite of what is needed.
    zone_e = ((ref <= 70) & (pred >= 180)) | ((ref >= 180) & (pred <= 70))

    # Zone D: failure to detect. Reference is out of range, prediction says fine.
    zone_d = (((ref < 70) & (pred > 70) & (pred < 180))
              | ((ref > 240) & (pred > 70) & (pred < 180)))

    # Zone C: overcorrection — prediction pushes treatment on a value in range.
    zone_c = (((ref >= 70) & (ref <= 180) & (pred > 180))
              | ((ref >= 70) & (ref <= 180) & (pred < 70)))

    zones[zone_c] = 2
    zones[zone_d] = 3
    zones[zone_e] = 4
    zones[zone_a] = 0     # zone A wins over the others by definition
    return zones


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Regression accuracy plus hypo/hyper alarm behaviour, in one dict."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true

    out = {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "mard": float(np.mean(np.abs(err) / y_true) * 100),
        "bias": float(np.mean(err)),
    }

    # --- hypoglycaemia alarm at the 30-minute horizon -------------------------
    actual_low = y_true < HYPO_THRESHOLD
    pred_low = y_pred < HYPO_THRESHOLD
    tp = float(np.sum(actual_low & pred_low))
    fp = float(np.sum(~actual_low & pred_low))
    fn = float(np.sum(actual_low & ~pred_low))

    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    out["hypo_recall"] = recall
    out["hypo_precision"] = precision
    out["hypo_f1"] = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    # False alarms per 24 h of continuous wear, at one prediction every 5 min.
    n_days = len(y_true) * 5 / (60 * 24)
    out["hypo_false_alarms_per_day"] = fp / n_days if n_days else 0.0

    # --- hyperglycaemia, same idea -------------------------------------------
    actual_high = y_true > HYPER_THRESHOLD
    pred_high = y_pred > HYPER_THRESHOLD
    tp_h = float(np.sum(actual_high & pred_high))
    fn_h = float(np.sum(actual_high & ~pred_high))
    out["hyper_recall"] = tp_h / (tp_h + fn_h) if tp_h + fn_h else 0.0

    # --- clinical acceptability ----------------------------------------------
    zones = _clarke_zones(y_true, y_pred)
    out["clarke_a"] = float(np.mean(zones == 0) * 100)
    out["clarke_ab"] = float(np.mean(zones <= 1) * 100)
    out["clarke_de"] = float(np.mean(zones >= 3) * 100)

    # RMSE restricted to windows that actually end low — where a glucose
    # forecaster earns its keep, and where the overall average hides failure.
    if actual_low.any():
        out["rmse_hypo"] = float(np.sqrt(np.mean(err[actual_low] ** 2)))
    else:
        out["rmse_hypo"] = float("nan")
    return out


HEADLINE = ["rmse", "mae", "mard", "rmse_hypo", "hypo_recall", "hypo_precision", "clarke_ab"]


def format_row(name: str, m: dict[str, float]) -> str:
    return (
        f"{name:<22s} {m['rmse']:7.2f} {m['mae']:7.2f} {m['mard']:6.2f}% "
        f"{m['rmse_hypo']:8.2f} {m['hypo_recall']:8.1%} {m['hypo_precision']:9.1%} "
        f"{m['clarke_ab']:7.2f}%"
    )


HEADER = (
    f"{'model':<22s} {'RMSE':>7s} {'MAE':>7s} {'MARD':>7s} "
    f"{'RMSE_hyp':>8s} {'recall':>8s} {'precision':>9s} {'Clarke_AB':>8s}"
)
