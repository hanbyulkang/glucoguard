"""Set each wearer's alarm threshold from their own first weeks of data.

Two independent population shifts in this project broke the alarm threshold and
neither broke the model's ranking. A cutoff tuned on one group of people lands
at 1.8 false alarms a day for one population and 14.7 for another, because
people differ enormously in how often they actually go low. Chasing a single
population-wide number is chasing something that does not exist.

The fix does not need a better model. It needs the threshold to belong to the
person: hold out a wearer's first `warmup_days` of readings, tune their cutoff
on that, and use it for the rest of their life with the device. This is exactly
what a real deployment can do — a CGM is worn continuously, so the warm-up data
arrives for free in the first two weeks.

Nothing here is fitted on the evaluation period, and no labels from it are used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.alarm import PREDICTIONS_PER_DAY, alarm_metrics, tune_threshold
from src.config import HYPO_THRESHOLD, SAMPLE_MINUTES

MIN_WARMUP_LOWS = 20        # below this the warm-up cannot pin a threshold down


@dataclass
class PatientCalibration:
    patient_id: str
    threshold: float
    fell_back: bool          # warm-up too sparse; population threshold used
    warmup_windows: int
    warmup_lows: int
    eval_windows: int


def split_by_time(times: np.ndarray, warmup_days: float) -> np.ndarray:
    """Boolean mask selecting the first `warmup_days` of a patient's record.

    The split is chronological, never random: a deployment only ever has the
    past to calibrate on, and shuffling would leak later behaviour into the
    threshold.
    """
    t = np.asarray(times, dtype="datetime64[ns]")
    cutoff = t.min() + np.timedelta64(int(warmup_days * 24 * 60), "m")
    return t < cutoff


def calibrate_patient(
    y: np.ndarray,
    score: np.ndarray,
    times: np.ndarray,
    patient_id: str,
    target_fa_per_day: float,
    population_threshold: float,
    warmup_days: float = 14.0,
) -> tuple[PatientCalibration, np.ndarray]:
    """Fit one wearer's cutoff on their warm-up; return it and the eval mask."""
    warmup = split_by_time(times, warmup_days)
    evaluation = ~warmup
    warmup_lows = int((y[warmup] < HYPO_THRESHOLD).sum())

    if warmup.sum() == 0 or warmup_lows < MIN_WARMUP_LOWS:
        # Not enough lows to see. Falling back is the honest behaviour — a
        # device should not invent a personal threshold from three events.
        cal = PatientCalibration(patient_id, population_threshold, True,
                                 int(warmup.sum()), warmup_lows,
                                 int(evaluation.sum()))
        return cal, evaluation

    threshold = tune_threshold(y[warmup], score[warmup], target_fa_per_day)
    cal = PatientCalibration(patient_id, threshold, False, int(warmup.sum()),
                             warmup_lows, int(evaluation.sum()))
    return cal, evaluation


def evaluate_strategies(
    y: np.ndarray,
    score: np.ndarray,
    times: np.ndarray,
    patient_ids: np.ndarray,
    population_threshold: float,
    target_fa_per_day: float,
    warmup_days: float = 14.0,
) -> dict:
    """Compare one shared cutoff against a per-wearer cutoff, same eval windows.

    Both strategies are scored on identical windows — everything after each
    patient's warm-up — so the comparison isolates the threshold and nothing
    else.
    """
    per_patient, pooled = [], {"pop": [], "cal": [], "y": []}

    for pid in sorted(set(patient_ids)):
        sel = patient_ids == pid
        cal, eval_mask = calibrate_patient(
            y[sel], score[sel], times[sel], pid,
            target_fa_per_day, population_threshold, warmup_days,
        )
        if eval_mask.sum() == 0:
            continue

        y_eval = y[sel][eval_mask]
        s_eval = score[sel][eval_mask]
        m_pop = alarm_metrics(y_eval, s_eval, population_threshold)
        m_cal = alarm_metrics(y_eval, s_eval, cal.threshold)

        per_patient.append({
            "patient_id": pid,
            "fell_back": cal.fell_back,
            "warmup_lows": cal.warmup_lows,
            "eval_windows": int(eval_mask.sum()),
            "population": m_pop,
            "calibrated": m_cal,
        })
        pooled["y"].append(y_eval)
        pooled["pop"].append(s_eval >= population_threshold)
        pooled["cal"].append(s_eval >= cal.threshold)

    y_all = np.concatenate(pooled["y"])
    out = {"target_fa_per_day": target_fa_per_day, "warmup_days": warmup_days,
           "patients": per_patient}

    for key, label in [("pop", "population"), ("cal", "calibrated")]:
        alarm = np.concatenate(pooled[key])
        low = y_all < HYPO_THRESHOLD
        tp, fp = float((alarm & low).sum()), float((alarm & ~low).sum())
        fn = float((~alarm & low).sum())
        days = len(y_all) * SAMPLE_MINUTES / (60 * 24)
        out[label] = {
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "false_alarms_per_day": fp / days,
        }

    # How reliably does each strategy actually hit the requested rate? This is
    # the number the whole idea lives or dies on.
    for key, label in [("population", "population"), ("calibrated", "calibrated")]:
        rates = np.array([p[key]["false_alarms_per_day"] for p in per_patient])
        out[label]["per_patient_fa"] = {
            "median": float(np.median(rates)),
            "iqr": [float(np.percentile(rates, 25)), float(np.percentile(rates, 75))],
            "min": float(rates.min()),
            "max": float(rates.max()),
            "within_2x_of_target": float(
                np.mean((rates >= target_fa_per_day / 2)
                        & (rates <= target_fa_per_day * 2))
            ),
        }
    return out
