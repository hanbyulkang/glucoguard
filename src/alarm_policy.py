"""Score the alarm the way a wearer experiences it, and keep it calibrated.

Everything up to now counted each five-minute reading as its own alarm
opportunity. That is not what a device does and not what a person feels. Under
per-reading accounting a single half-hour of nuisance alarming counts as six
false alarms, and one low that the model misses at onset but catches two
readings later counts as several misses and several hits at once.

A real alarm fires, then goes quiet for a while. A wearer counts events: how
many times did it interrupt me today, and did it warn me before each low. This
module implements that.

It also replaces fit-once calibration with a rolling one. Fitting a threshold on
someone's first fortnight and never touching it again happened to hold up here,
but it is a fragile design: it cannot follow a person whose control changes, and
"it held on our data" is not a safety argument. Re-fitting on a trailing window
costs nothing, the data is already there, and tracks the wearer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.alarm import tune_threshold
from src.config import HYPO_THRESHOLD, SAMPLE_MINUTES

REFRACTORY_MINUTES = 30      # how long an alarm stays quiet after firing
WARNING_WINDOW_MINUTES = 60  # how early a warning still counts as having warned


def dedupe_alarms(raised: np.ndarray,
                  refractory_minutes: int = REFRACTORY_MINUTES) -> np.ndarray:
    """Collapse a run of consecutive alarm conditions into discrete events.

    Returns a mask marking only the readings where the device would actually
    make a sound: the first one, then nothing for the refractory period even if
    the condition persists.
    """
    steps = max(1, int(round(refractory_minutes / SAMPLE_MINUTES)))
    fired = np.zeros_like(raised, dtype=bool)
    quiet_until = -1
    for i, on in enumerate(raised):
        if on and i > quiet_until:
            fired[i] = True
            quiet_until = i + steps
    return fired


def hypo_episodes(y: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges of each continuous stretch below the hypo threshold."""
    low = y < HYPO_THRESHOLD
    if not low.any():
        return []
    edges = np.flatnonzero(np.diff(low.astype(np.int8)))
    starts = np.r_[0, edges + 1][low[np.r_[0, edges + 1]]]
    ends = np.r_[edges, len(low) - 1][low[np.r_[edges, len(low) - 1]]]
    return list(zip(starts.tolist(), ends.tolist()))


@dataclass
class EventMetrics:
    episodes: int
    episodes_warned: int
    episode_recall: float
    alarm_events: int
    false_alarm_events: int
    false_alarms_per_day: float
    alarms_per_day: float
    days: float
    median_lead_minutes: float


def event_metrics(y: np.ndarray, alarm_raised: np.ndarray,
                  refractory_minutes: int = REFRACTORY_MINUTES,
                  warning_window_minutes: int = WARNING_WINDOW_MINUTES) -> EventMetrics:
    """Episode-level recall and event-level false alarms.

    An episode counts as warned if the device made a sound at any point in the
    hour before glucose crossed the threshold, the forecast is issued 30
    minutes ahead, so a warning can legitimately arrive a little earlier than
    that if the model saw the fall developing.

    An alarm event counts as false if no low began within the warning window
    after it. A burst that correctly precedes a low is one true alarm, not six.
    """
    fired = dedupe_alarms(alarm_raised, refractory_minutes)
    fired_idx = np.flatnonzero(fired)
    episodes = hypo_episodes(y)
    lookback = max(1, int(round(warning_window_minutes / SAMPLE_MINUTES)))

    warned, leads = 0, []
    useful = np.zeros(len(fired_idx), dtype=bool)
    claimed = np.zeros(len(fired_idx), dtype=bool)

    # Episodes are walked in order and each alarm can be credited to at most one
    # of them. Without this a single alarm sitting between two nearby lows would
    # be counted as having warned about both, inflating recall for free.
    for start, _end in episodes:
        window = (fired_idx >= start - lookback) & (fired_idx <= start) & ~claimed
        if window.any():
            warned += 1
            useful |= window
            claimed |= window
            first = fired_idx[window][0]
            leads.append((start - first) * SAMPLE_MINUTES)

    # Alarms inside an ongoing episode are not false, the wearer is low and the
    # device is right to be noisy, they simply do not earn a fresh warning.
    inside = np.zeros(len(y), dtype=bool)
    for start, end in episodes:
        inside[start : end + 1] = True
    useful |= inside[fired_idx]

    days = len(y) * SAMPLE_MINUTES / (60 * 24)
    false_events = int((~useful).sum())
    return EventMetrics(
        episodes=len(episodes),
        episodes_warned=warned,
        episode_recall=warned / len(episodes) if episodes else float("nan"),
        alarm_events=int(fired.sum()),
        false_alarm_events=false_events,
        false_alarms_per_day=false_events / days if days else float("nan"),
        alarms_per_day=int(fired.sum()) / days if days else float("nan"),
        days=days,
        median_lead_minutes=float(np.median(leads)) if leads else float("nan"),
    )


def tune_event_threshold(y: np.ndarray, score: np.ndarray,
                         target_false_events_per_day: float,
                         refractory_minutes: int = REFRACTORY_MINUTES) -> float:
    """Find the cutoff whose *de-duplicated* false alarms hit the budget.

    Tuning on per-reading counts and then de-duplicating would land far below
    the budget, because de-duplication removes most of what the tuner was
    counting. The budget has to be expressed in the same units the wearer
    experiences.
    """
    candidates = np.unique(np.quantile(score, np.linspace(0.80, 0.99995, 160)))
    best, best_gap = candidates[-1], float("inf")
    for thr in candidates:
        m = event_metrics(y, score >= thr, refractory_minutes)
        if not np.isfinite(m.false_alarms_per_day):
            continue
        # Prefer the most sensitive threshold that stays inside the budget;
        # if none does, take the one that comes closest from above.
        if m.false_alarms_per_day <= target_false_events_per_day:
            return float(thr)
        gap = m.false_alarms_per_day - target_false_events_per_day
        if gap < best_gap:
            best, best_gap = thr, gap
    return float(best)


def rolling_thresholds(y: np.ndarray, score: np.ndarray, times: np.ndarray,
                       target_false_events_per_day: float,
                       trailing_days: float = 28.0,
                       refit_days: float = 7.0,
                       min_lows: int = 15) -> tuple[np.ndarray, dict]:
    """Re-fit the cutoff every `refit_days` on the trailing window.

    Returns a per-reading threshold array, so every prediction is scored against
    a cutoff derived only from data strictly before it. Where the trailing
    window holds too few lows to fit on, the previous threshold carries forward.
    """
    t = np.asarray(times, dtype="datetime64[ns]")
    order = np.argsort(t)
    t_sorted, y_sorted, s_sorted = t[order], y[order], score[order]

    out = np.full(len(y), np.nan)
    start = t_sorted[0]
    refit = np.timedelta64(int(refit_days * 24 * 60), "m")
    trailing = np.timedelta64(int(trailing_days * 24 * 60), "m")

    current: float | None = None
    n_refits, n_carried = 0, 0
    edge = start + trailing
    schedule: list[dict] = []

    while edge <= t_sorted[-1] + refit:
        window = (t_sorted >= edge - trailing) & (t_sorted < edge)
        lows = int((y_sorted[window] < HYPO_THRESHOLD).sum())
        refitted = False
        if window.sum() > 0 and lows >= min_lows:
            current = tune_event_threshold(
                y_sorted[window], s_sorted[window], target_false_events_per_day
            )
            n_refits += 1
            refitted = True
        elif current is not None:
            n_carried += 1

        applies = (t_sorted >= edge) & (t_sorted < edge + refit)
        if current is not None:
            out[order[applies]] = current
            schedule.append({
                "date": str(np.datetime_as_string(edge, unit="D")),
                "days_since_start": float((edge - start) / np.timedelta64(1, "D")),
                "threshold": float(current),
                "refitted": refitted,
                # The wearer's own recent behaviour, which is what moves the
                # threshold. Reported alongside it so the two can be compared.
                "trailing_hypo_rate": float((y_sorted[window] < HYPO_THRESHOLD).mean())
                if window.sum() else float("nan"),
                "trailing_lows": lows,
            })
        edge = edge + refit

    return out, {"refits": n_refits, "carried_forward": n_carried,
                 "covered": float(np.isfinite(out).mean()),
                 "schedule": schedule}
