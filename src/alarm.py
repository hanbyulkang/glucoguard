"""Turning a forecast into a low-glucose alarm — and comparing alarms fairly.

The first version of this project scored the alarm by asking whether the point
forecast happened to land below 70 mg/dL. That comparison is rigged, and it
produced a genuinely misleading result: the network had *worse* recall than
persistence, which looked like the network was worse at seeing lows.

It is not. A forecast trained on squared error is pulled toward the mean, so it
under-shoots the rare extremes; persistence has no such pull and linear
extrapolation actively over-shoots the fall. Reading a single fixed cutoff on
three differently-biased predictors compares their biases, not their skill.

The fix is to treat the alarm as a decision with a free parameter. Every model
emits a *score* — how much it believes a low is coming — and the threshold on
that score is tuned, on validation, to a chosen false-alarm budget. Then all
models are compared at the same budget, which is the only comparison a patient
would recognise: *given that I will tolerate this many false alarms per day,
how many of my lows does each system catch?*
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from src.config import HYPO_THRESHOLD, SAMPLE_MINUTES

# A day of continuous wear at one prediction per sample.
PREDICTIONS_PER_DAY = 24 * 60 / SAMPLE_MINUTES


def risk_score(pred: np.ndarray, sigma: np.ndarray | None = None) -> np.ndarray:
    """Convert a forecast into "how much do I believe a low is coming".

    For a point forecast this is just how far below the threshold it sits, so
    thresholding the score is equivalent to thresholding the prediction — but
    now with a tunable offset rather than a hard-coded 70.

    For a probabilistic forecast it is the actual probability that glucose ends
    up below the threshold, which is the quantity the decision needs. A wide
    predictive distribution centred at 85 can carry more low-risk than a narrow
    one centred at 78, and only the probabilistic score can express that.
    """
    if sigma is None:
        return HYPO_THRESHOLD - pred
    # P(y < threshold) under a Gaussian predictive distribution.
    return ndtr((HYPO_THRESHOLD - pred) / np.maximum(sigma, 1e-6))


def false_alarms_per_day(y_true: np.ndarray, alarm: np.ndarray) -> float:
    fp = float(np.sum(alarm & (y_true >= HYPO_THRESHOLD)))
    return fp / (len(y_true) / PREDICTIONS_PER_DAY)


def tune_threshold(
    y_true: np.ndarray, score: np.ndarray, budget_per_day: float
) -> float:
    """Highest-recall threshold that stays inside a false-alarm budget.

    Tuned on validation only. Returns the score cutoff; alarm when
    ``score >= cutoff``.
    """
    order = np.argsort(-score)                    # most alarming first
    is_fp = (y_true[order] >= HYPO_THRESHOLD).astype(np.int64)
    cum_fp = np.cumsum(is_fp)

    n_days = len(y_true) / PREDICTIONS_PER_DAY
    allowed = budget_per_day * n_days
    k = int(np.searchsorted(cum_fp, allowed, side="right"))
    if k == 0:
        return float(score[order[0]] + 1e-9)      # budget too tight to alarm at all
    return float(score[order[min(k, len(order)) - 1]])


def alarm_metrics(
    y_true: np.ndarray, score: np.ndarray, threshold: float
) -> dict[str, float]:
    alarm = score >= threshold
    actual_low = y_true < HYPO_THRESHOLD

    tp = float(np.sum(alarm & actual_low))
    fp = float(np.sum(alarm & ~actual_low))
    fn = float(np.sum(~alarm & actual_low))

    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "threshold": threshold,
        "recall": recall,
        "precision": precision,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_alarms_per_day": false_alarms_per_day(y_true, alarm),
    }


def pr_curve(y_true: np.ndarray, score: np.ndarray, n_points: int = 200) -> dict:
    """Recall and false-alarm rate across the whole threshold range.

    This is the object that actually answers "which alarm is better" — a single
    (recall, precision) pair is one arbitrary point on it.
    """
    order = np.argsort(-score)
    actual_low = (y_true[order] < HYPO_THRESHOLD)
    tp = np.cumsum(actual_low)
    fp = np.cumsum(~actual_low)
    total_low = max(int(actual_low.sum()), 1)
    n_days = len(y_true) / PREDICTIONS_PER_DAY

    idx = np.unique(np.linspace(0, len(order) - 1, n_points).astype(int))
    return {
        "recall": (tp[idx] / total_low).tolist(),
        "precision": (tp[idx] / np.maximum(tp[idx] + fp[idx], 1)).tolist(),
        "false_alarms_per_day": (fp[idx] / n_days).tolist(),
        "threshold": score[order][idx].tolist(),
    }


def recall_at_budget(
    y_val: np.ndarray, score_val: np.ndarray,
    y_test: np.ndarray, score_test: np.ndarray,
    budgets: tuple[float, ...] = (1.0, 3.0, 6.0),
) -> dict[str, dict]:
    """Tune on validation at each budget, report on test. The headline table."""
    out = {}
    for b in budgets:
        thr = tune_threshold(y_val, score_val, b)
        out[f"{b:g}/day"] = alarm_metrics(y_test, score_test, thr)
    return out
