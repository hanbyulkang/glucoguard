"""Live CGM input and phone alerts.

Two sources, because a demo should not depend on someone being hypoglycaemic on
cue:

* **Nightscout** — the self-hosted server most of this community already runs,
  and the same software that produced the training archive. A URL is all that is
  needed; the read API is public on most instances.
* **Replay** — a recorded trace stepped forward in real time. Identical code
  path, no credentials, and you can point it at a day that actually contains a
  low.

Alerts go out over `ntfy.sh`, a free publish-subscribe service that needs no
account: you pick a topic string, subscribe on your phone, and anything posted
to that topic arrives as a push notification. The topic is supplied by the user
at runtime and never stored.

None of this is a medical device. It is a demonstration that the forecast can
be driven by a live feed and reach a phone.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.config import (
    GLUCOSE_MAX,
    GLUCOSE_MIN,
    HISTORY_STEPS,
    HORIZON_MINUTES,
    HYPO_THRESHOLD,
    SAMPLE_MINUTES,
)

USER_AGENT = "GlucoGuard/0.1 (research demo)"
TIMEOUT = 10


# --------------------------------------------------------------------------- #
# Nightscout
# --------------------------------------------------------------------------- #
def fetch_nightscout(base_url: str, count: int = 64,
                     token: str | None = None) -> pd.DataFrame:
    """Pull the most recent sensor readings from a Nightscout instance.

    Raises on network or parse failure rather than returning something
    plausible-looking — a glucose display that silently invents data is worse
    than one that says it is broken.
    """
    base = base_url.strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        base = "https://" + base

    params = {"count": str(count), "find[type]": "sgv"}
    if token:
        params["token"] = token
    url = f"{base}/api/v1/entries.json?{urllib.parse.urlencode(params)}"

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    if not isinstance(payload, list):
        raise ValueError("Nightscout returned something that is not a list of entries")

    rows = []
    for rec in payload:
        if not isinstance(rec, dict):
            continue
        sgv, date = rec.get("sgv"), rec.get("date")
        if sgv is None or date is None:
            continue
        try:
            sgv, date = float(sgv), int(date)
        except (TypeError, ValueError):
            continue
        if GLUCOSE_MIN <= sgv <= GLUCOSE_MAX:
            rows.append((pd.to_datetime(date, unit="ms", utc=True), sgv))

    if not rows:
        raise ValueError("no usable sensor readings in the response")

    df = pd.DataFrame(rows, columns=["datetime", "glucose"])
    return df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# window assembly
# --------------------------------------------------------------------------- #
@dataclass
class LiveWindow:
    values: np.ndarray | None       # (HISTORY_STEPS,) mg/dL, or None if unusable
    last_time: pd.Timestamp | None
    reason: str = ""                # why it is unusable, when it is


def to_window(readings: pd.DataFrame, now: pd.Timestamp | None = None) -> LiveWindow:
    """Snap recent readings onto the model's grid, or explain why we cannot.

    Refusing is a feature. Real feeds drop out, and a forecast built from a
    two-hour hole filled with interpolation would be a confident invention. The
    caller is expected to show the reason rather than a number.
    """
    if readings.empty:
        return LiveWindow(None, None, "no readings received")

    freq = f"{SAMPLE_MINUTES}min"
    df = readings.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.round(freq)
    df = df.groupby("datetime", as_index=False)["glucose"].median()

    end = (pd.to_datetime(now, utc=True).round(freq) if now is not None
           else df["datetime"].max())
    start = end - pd.Timedelta(minutes=SAMPLE_MINUTES * (HISTORY_STEPS - 1))
    grid = pd.date_range(start, end, freq=freq)

    series = df.set_index("datetime")["glucose"].reindex(grid)
    missing = int(series.isna().sum())
    if missing:
        # Short gaps are fair to bridge; the same 15-minute rule the training
        # windows used. Anything longer and we decline.
        series = series.interpolate(limit=3, limit_area="inside")
    if series.isna().any():
        return LiveWindow(None, None,
                          f"gap in the last {HISTORY_STEPS * SAMPLE_MINUTES} minutes "
                          f"too long to bridge ({missing} of {HISTORY_STEPS} samples missing)")

    age = pd.Timestamp.now(tz=timezone.utc) - end
    if age > timedelta(minutes=20):
        return LiveWindow(None, None,
                          f"most recent reading is {age.total_seconds() / 60:.0f} minutes old")

    return LiveWindow(series.to_numpy(dtype=np.float32), end)


def replay_window(series: pd.DataFrame, at: pd.Timestamp) -> LiveWindow:
    """The same assembly, against a recorded trace, as if `at` were now."""
    end = pd.to_datetime(at, utc=True)
    start = end - pd.Timedelta(minutes=SAMPLE_MINUTES * (HISTORY_STEPS - 1))
    df = series.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    recent = df[(df["datetime"] >= start) & (df["datetime"] <= end)]

    if len(recent) < HISTORY_STEPS or recent["glucose"].isna().any():
        return LiveWindow(None, None, "recorded trace has a gap at this point")
    return LiveWindow(recent["glucose"].to_numpy(dtype=np.float32)[-HISTORY_STEPS:], end)


# --------------------------------------------------------------------------- #
# push notification
# --------------------------------------------------------------------------- #
def send_alert(topic: str, glucose_now: float, risk: float,
               predicted: float, dry_run: bool = False) -> str:
    """Post a low-glucose warning to an ntfy.sh topic. Returns a status string."""
    topic = topic.strip().strip("/")
    if not topic:
        return "no topic set"

    title = f"Glucose heading low — {HORIZON_MINUTES} min warning"
    body = (
        f"Now {glucose_now:.0f} mg/dL, predicted {predicted:.0f} mg/dL "
        f"in {HORIZON_MINUTES} minutes ({risk:.0%} chance of going under "
        f"{HYPO_THRESHOLD}).\n\n"
        f"Research demo. Not a medical device — do not treat based on this."
    )
    if dry_run:
        return f"(dry run) would send to ntfy.sh/{topic}"

    req = urllib.request.Request(
        f"https://ntfy.sh/{urllib.parse.quote(topic)}",
        data=body.encode("utf-8"),
        headers={
            "User-Agent": USER_AGENT,
            "Title": title,
            "Priority": "high",
            "Tags": "warning",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return f"sent ({resp.status})"
    except (urllib.error.URLError, OSError) as exc:
        return f"failed: {exc}"
