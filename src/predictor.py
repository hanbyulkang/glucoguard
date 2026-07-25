"""Load a trained checkpoint and serve predictions. Used by the demo app."""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr

from src.config import (
    ARTIFACTS_DIR,
    CACHE_DIR,
    HISTORY_STEPS,
    HORIZON_STEPS,
    HYPO_THRESHOLD,
    SAMPLE_MINUTES,
)
from src.models.nets import build


class Forecaster:
    """Thin wrapper: raw mg/dL windows in, forecast (and risk) out."""

    def __init__(self, checkpoint: str):
        blob = torch.load(ARTIFACTS_DIR / f"{checkpoint}.pt", map_location="cpu",
                          weights_only=False)
        cfg, norm = blob["config"], blob["norm"]
        self.name = checkpoint
        self.arch = cfg["model"]
        self.n_params = blob["n_params"]
        self.probabilistic = bool(cfg.get("probabilistic", False))
        self.classifies = bool(cfg.get("classify", False))
        self.model = build(
            cfg["model"], mean=norm["mean"], std=norm["std"],
            heteroscedastic=self.probabilistic, classify=self.classifies,
        )
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def _raw(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32))
        if x.ndim == 1:
            x = x.unsqueeze(0)
        out = []
        for i in range(0, len(x), 8192):
            out.append(self.model(x[i : i + 8192]).numpy())
        return np.concatenate(out)

    def predict(self, windows: np.ndarray) -> np.ndarray:
        """Point forecast in mg/dL, whatever heads the checkpoint carries."""
        raw = self._raw(windows)
        return raw[..., 0] if raw.ndim == 2 else raw

    def predict_full(self, windows: np.ndarray) -> dict[str, np.ndarray | None]:
        """Forecast plus, where the model has them, spread and low-risk.

        ``hypo_prob`` prefers the trained classifier over the Gaussian tail:
        the classifier was optimised for this exact decision, while the Gaussian
        probability is a by-product of a model optimised to be accurate on
        average.
        """
        raw = self._raw(windows)
        if raw.ndim == 1:
            return {"mu": raw, "sigma": None, "hypo_prob": None}

        mu = raw[..., 0]
        sigma = raw[..., 1] if self.probabilistic else None
        if self.classifies:
            hypo_prob = 1.0 / (1.0 + np.exp(-raw[..., -1]))
        elif sigma is not None:
            hypo_prob = ndtr((HYPO_THRESHOLD - mu) / np.maximum(sigma, 1e-6))
        else:
            hypo_prob = None
        return {"mu": mu, "sigma": sigma, "hypo_prob": hypo_prob}


def available_checkpoints() -> list[str]:
    return sorted(p.stem for p in ARTIFACTS_DIR.glob("*.pt") if p.stem != "smoke")


def best_checkpoint() -> str | None:
    """Whichever model the sweep selected on validation, if it is on disk."""
    sweep = ARTIFACTS_DIR / "sweep.json"
    names = available_checkpoints()
    if sweep.exists():
        selected = json.loads(sweep.read_text()).get("selected_on_validation")
        if selected in names:
            return selected
    return names[0] if names else None


@lru_cache(maxsize=1)
def load_cgm() -> pd.DataFrame:
    return pd.read_parquet(CACHE_DIR / "cgm.parquet")


@lru_cache(maxsize=1)
def load_splits() -> dict:
    return json.loads((ARTIFACTS_DIR / "splits.json").read_text())


def patient_series(patient_id: str) -> pd.DataFrame:
    df = load_cgm()
    out = df[df["patient_id"] == patient_id].sort_values("datetime")
    return out.reset_index(drop=True)


def rolling_forecast(
    series: pd.DataFrame, forecaster: Forecaster
) -> pd.DataFrame:
    """Predict at every step of a slice where the full history is available.

    Returns one row per prediction: the time the forecast is *for*, the
    prediction, and the value that actually occurred.
    """
    values = series["glucose"].to_numpy(dtype=np.float32)
    times = series["datetime"].to_numpy()
    span = HISTORY_STEPS + HORIZON_STEPS
    if len(values) < span:
        return pd.DataFrame(columns=["issued_at", "target_time", "predicted", "actual"])

    n = len(values) - span + 1
    hist_idx = np.arange(HISTORY_STEPS)[None, :] + np.arange(n)[:, None]
    tgt_idx = np.arange(n) + span - 1

    windows = values[hist_idx]
    # Predict only where the full history exists AND the outcome was really
    # measured. Scoring against an interpolated target would flatter the model.
    valid = ~np.isnan(windows).any(axis=1) & ~np.isnan(values[tgt_idx])
    if not valid.any():
        return pd.DataFrame(columns=["issued_at", "target_time", "predicted", "actual"])

    out = forecaster.predict_full(windows[valid])
    frame = pd.DataFrame({
        "issued_at": times[hist_idx[valid][:, -1]],
        "target_time": times[tgt_idx[valid]],
        "predicted": out["mu"],
        "actual": values[tgt_idx[valid]],
        "current": windows[valid][:, -1],
    })
    if out["sigma"] is not None:
        frame["sigma"] = out["sigma"]
    if out["hypo_prob"] is not None:
        frame["hypo_prob"] = out["hypo_prob"]
    return frame


def alarm_flags(frame: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
    """Which rows raise a low-glucose alarm.

    Prefers the model's own low-risk output over "did the point forecast dip
    under 70". Thresholding the forecast makes the alarm a side effect of a
    number optimised for average accuracy; the risk head is optimised for this
    decision, and its cutoff is a tunable knob rather than a fixed constant.
    """
    if "hypo_prob" in frame.columns:
        return (frame["hypo_prob"] >= (0.5 if threshold is None else threshold)).to_numpy()
    cutoff = HYPO_THRESHOLD if threshold is None else threshold
    return (frame["predicted"] < cutoff).to_numpy()


def hypo_episodes(frame: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Find each contiguous run below 70 mg/dL and how early it was called.

    Per-reading recall overstates usefulness: one long low counts many times
    over. What a patient experiences is an *episode*, and what matters is
    whether any warning arrived before it started, and how far ahead.

    ``lead_minutes`` is measured from the moment the earliest correct warning
    was issued to the moment glucose actually crossed the threshold. An episode
    that was never called gets NaN.
    """
    low = (frame["actual"] < HYPO_THRESHOLD).to_numpy()
    if not low.any():
        return pd.DataFrame(columns=["onset", "duration_min", "lead_minutes"])

    onset_time = pd.to_datetime(frame["target_time"]).to_numpy()
    issued = pd.to_datetime(frame["issued_at"]).to_numpy()
    called = alarm_flags(frame, threshold)

    # Boundaries of each run of consecutive low readings.
    edges = np.flatnonzero(np.diff(low.astype(np.int8)))
    starts = np.r_[0, edges + 1][low[np.r_[0, edges + 1]]]
    ends = np.r_[edges, len(low) - 1][low[np.r_[edges, len(low) - 1]]]

    rows = []
    for s, e in zip(starts, ends):
        onset = onset_time[s]
        # Row s is the forecast that targets the onset reading itself; it was
        # issued one horizon earlier. An episode counts as caught only if that
        # forecast called it — a warning that lands after glucose has already
        # dropped is not a warning. Walking back through the unbroken run of
        # earlier low calls gives credit when the model saw it coming sooner.
        if called[s]:
            j = s
            while j > 0 and called[j - 1]:
                j -= 1
            lead = (onset - issued[j]) / np.timedelta64(1, "m")
        else:
            lead = np.nan
        rows.append({
            "onset": onset,
            "duration_min": (e - s + 1) * SAMPLE_MINUTES,
            "lead_minutes": lead,
        })
    return pd.DataFrame(rows)


def hypo_lead_time(
    frame: pd.DataFrame, threshold: float | None = None
) -> tuple[float | None, float]:
    """Median warning in minutes, and the share of episodes caught at all."""
    ep = hypo_episodes(frame, threshold)
    if ep.empty:
        return None, 0.0
    caught = ep["lead_minutes"].notna()
    median = float(ep.loc[caught, "lead_minutes"].median()) if caught.any() else None
    return median, float(caught.mean())
