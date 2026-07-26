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

DEMO_BUNDLE = ARTIFACTS_DIR.parent / "demo_data"


def demo_mode() -> bool:
    """True when only the small shipped bundle is present.

    A public deploy carries 4 MB of precomputed forecasts instead of the 110 MB
    working set, so the app has to run without the raw archive or the full
    cache. Detecting it here keeps every caller from having to know.
    """
    return not (CACHE_DIR / "cgm.parquet").exists() and DEMO_BUNDLE.exists()




def _device() -> torch.device:
    """Use the GPU when there is one. Inference over a full patient record is
    ~100k windows, and on CPU that is slow enough to be felt in the demo."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Forecaster:
    """Thin wrapper: raw mg/dL windows in, forecast (and risk) out."""

    def __init__(self, checkpoint: str):
        blob = torch.load(checkpoint_path(checkpoint), map_location="cpu",
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
        self.device = _device()
        self.model.to(self.device).eval()

    @torch.no_grad()
    def _raw(self, windows: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32))
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        out = []
        for i in range(0, len(x), 16384):
            out.append(self.model(x[i : i + 16384]).float().cpu().numpy())
        return np.concatenate(out)

    def predict(self, windows: np.ndarray) -> np.ndarray:
        """Point forecast in mg/dL, whatever heads the checkpoint carries."""
        raw = self._raw(windows)
        return raw[... 0] if raw.ndim == 2 else raw

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

        mu = raw[... 0]
        sigma = raw[... 1] if self.probabilistic else None
        if self.classifies:
            hypo_prob = 1.0 / (1.0 + np.exp(-raw[... -1]))
        elif sigma is not None:
            hypo_prob = ndtr((HYPO_THRESHOLD - mu) / np.maximum(sigma, 1e-6))
        else:
            hypo_prob = None
        return {"mu": mu, "sigma": sigma, "hypo_prob": hypo_prob}


@lru_cache(maxsize=1)
def bundle_summary() -> dict:
    """Per-wearer facts the deployed bundle carries instead of the raw table."""
    path = DEMO_BUNDLE / "summary.json"
    return json.loads(path.read_text()).get("wearers", {}) if path.exists() else {}


def wearer_facts(patient_id: str) -> dict:
    """Record length, reading count and time-below-70: however they are available.

    On a full checkout these come from the glucose table; on a deploy they come
    from the bundle, which stores them precisely so the raw table can be left
    out of the repository.
    """
    if demo_mode():
        s = bundle_summary().get(patient_id, {})
        return {"days": s.get("days", 0.0), "readings": s.get("windows", 0),
                "time_below_70": s.get("time_below_70", 0.0)}
    series = patient_series(patient_id)
    from src.config import SAMPLE_MINUTES as _sm
    return {
        "days": len(series) * _sm / (60 * 24),
        "readings": int(series["glucose"].notna().sum()),
        "time_below_70": float((series["glucose"] < HYPO_THRESHOLD).mean()),
    }


def checkpoint_path(name: str):
    """Where a checkpoint lives: artifacts normally, the bundle on a deploy."""
    local = ARTIFACTS_DIR / f"{name}.pt"
    return local if local.exists() else DEMO_BUNDLE / f"{name}.pt"


def available_checkpoints() -> list[str]:
    found = {p.stem for p in ARTIFACTS_DIR.glob("*.pt")}
    if DEMO_BUNDLE.exists():
        found |= {p.stem for p in DEMO_BUNDLE.glob("*.pt")}
    return sorted(n for n in found if n != "smoke")


def best_checkpoint() -> str | None:
    """The checkpoint the demo should open with.

    Deliberately ranks by *alarm* performance when that has been measured,
    not by RMSE. The two disagree: the lowest-RMSE model is the one most pulled
    toward the mean, and therefore the most reluctant to call the lows this
    product exists to catch. Choosing on RMSE would ship the wrong model.

    Falls back to validation RMSE, and then to whatever is on disk.
    """
    names = set(available_checkpoints())
    if not names:
        return None

    alarm = ARTIFACTS_DIR / "alarm.json"
    if alarm.exists():
        report = json.loads(alarm.read_text())
        ranked = [
            (max(b["recall"] for b in r["budgets"].values()), n)
            for n, r in report.items() if n in names
        ]
        if ranked:
            return max(ranked)[1]

    sweep = ARTIFACTS_DIR / "sweep.json"
    if sweep.exists():
        results = json.loads(sweep.read_text())["results"]
        on_disk = [r for r in results if r["name"] in names]
        if on_disk:
            return min(on_disk, key=lambda r: r["val"]["rmse"])["name"]

    return sorted(names)[0]


@lru_cache(maxsize=1)
def load_alarm_report() -> dict:
    path = ARTIFACTS_DIR / "alarm.json"
    return json.loads(path.read_text()) if path.exists() else {}


def alarm_budgets() -> list[str]:
    """False-alarm budgets the alarm report tuned a threshold for."""
    report = load_alarm_report()
    for entry in report.values():
        return list(entry.get("budgets", {}))
    return []


def tuned_threshold(forecaster: "Forecaster", budget: str) -> float | None:
    """The validation-tuned alarm cutoff, in the units ``alarm_flags`` expects.

    Without this the demo would fall back to an arbitrary cutoff, probability
    0.5, or 70 mg/dL on the point forecast, and show a far worse alarm than the
    one actually measured. The threshold is part of the system, not a detail.
    """
    entry = load_alarm_report().get(forecaster.name)
    if not entry:
        return None
    budgets = entry.get("budgets", {})
    chosen = budgets.get(budget) or next(iter(budgets.values()), None)
    if chosen is None:
        return None

    threshold = float(chosen["threshold"])
    # alarm.json stores the score the report thresholded. For a classifier that
    # score is a raw logit, while alarm_flags compares probabilities.
    if forecaster.classifies:
        return 1.0 / (1.0 + np.exp(-threshold))
    if forecaster.probabilistic:
        return threshold                      # already P(low)
    return HYPO_THRESHOLD - threshold         # score was (70 - prediction)


@lru_cache(maxsize=1)
def load_cgm() -> pd.DataFrame:
    if demo_mode():
        raise FileNotFoundError(
            "The raw glucose table is not part of the deployed bundle; the app "
            "reads precomputed forecasts instead."
        )
    return pd.read_parquet(CACHE_DIR / "cgm.parquet")


@lru_cache(maxsize=1)
def load_splits() -> dict:
    """Patient split, restricted to what the deploy actually carries."""
    path = ARTIFACTS_DIR / "splits.json"
    if not path.exists() and (DEMO_BUNDLE / "splits.json").exists():
        path = DEMO_BUNDLE / "splits.json"
    splits = json.loads(path.read_text())
    if demo_mode():
        shipped = {p.stem for p in (DEMO_BUNDLE / "forecasts").glob("*.parquet")}
        splits = {k: [p for p in v if p in shipped] if k == "test" else v
                  for k, v in splits.items()}
    return splits


def patient_series(patient_id: str) -> pd.DataFrame:
    df = load_cgm()
    out = df[df["patient_id"] == patient_id].sort_values("datetime")
    return out.reset_index(drop=True)


def forecast_cache_path(patient_id: str, model_name: str):
    if demo_mode():
        return DEMO_BUNDLE / "forecasts" / f"{patient_id}.parquet"
    return CACHE_DIR / "forecasts" / model_name / f"{patient_id}.parquet"


def cached_forecast(patient_id: str, forecaster: Forecaster) -> pd.DataFrame:
    """Rolling forecast for a whole patient, memoised on disk.

    A single patient is ~100k windows. Recomputing that on every interaction
    makes the demo feel broken, and re-running the model live adds nothing a
    viewer can see. Compute once, read thereafter.
    """
    path = forecast_cache_path(patient_id, forecaster.name)
    if path.exists():
        return pd.read_parquet(path)
    if demo_mode():
        # Nothing to fall back on: the bundle is the whole dataset here.
        return pd.DataFrame(columns=["issued_at", "target_time", "predicted",
                                     "actual", "current"])

    frame = rolling_forecast(patient_series(patient_id), forecaster)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


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
        # forecast called it, a warning that lands after glucose has already
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
