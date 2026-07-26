"""GlucoGuard live — drive the forecast from a running CGM feed.

Run:  streamlit run live_app.py

Two inputs. A Nightscout URL reads a real, currently-running sensor. Replay
steps a recorded trace forward in real time, which needs no credentials and can
be pointed at a day that actually contains a low.

The alarm threshold is the wearer's own, fitted on their first two weeks, for
the reason CALIBRATION.md sets out: a shared threshold means a different alarm
rate for every person.

This is a demonstration. It is not a medical device, it does not recommend
insulin, and nothing here should be treated as clinical advice.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.alarm import tune_threshold
from src.calibration import split_by_time
from src.config import (
    ARTIFACTS_DIR,
    HISTORY_STEPS,
    HORIZON_MINUTES,
    HYPER_THRESHOLD,
    HYPO_THRESHOLD,
    SAMPLE_MINUTES,
)
from src.live import fetch_nightscout, replay_window, send_alert, to_window
from src.predictor import (
    Forecaster,
    available_checkpoints,
    best_checkpoint,
    cached_forecast,
    load_splits,
    patient_series,
)
from src import ui
from src.theme import ACTUAL, CRITICAL, INK_MUTED, PREDICTED, WARNING, style


def as_utc(value) -> pd.Timestamp:
    """Coerce to a UTC timestamp whether or not it already carries a zone."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")



@st.cache_resource(show_spinner=False)
def get_forecaster(name: str) -> Forecaster:
    return Forecaster(name)


@st.cache_data(show_spinner=False)
def personal_threshold(patient_id: str, model_name: str,
                       target_fa: float, warmup_days: float) -> dict:
    """Fit this wearer's cutoff on their own first weeks, as a device would."""
    frame = cached_forecast(patient_id, get_forecaster(model_name))
    if "hypo_prob" not in frame.columns or frame.empty:
        return {"threshold": None, "reason": "model has no risk output"}

    warmup = split_by_time(frame["target_time"].to_numpy(), warmup_days)
    lows = int((frame.loc[warmup, "actual"] < HYPO_THRESHOLD).sum())
    if lows < 20:
        return {"threshold": None, "reason": f"only {lows} lows in the warm-up"}

    thr = tune_threshold(frame.loc[warmup, "actual"].to_numpy(),
                         frame.loc[warmup, "hypo_prob"].to_numpy(), target_fa)
    return {"threshold": float(thr), "lows": lows,
            "warmup_windows": int(warmup.sum())}


checkpoints = available_checkpoints()
if not checkpoints:
    st.error("No trained model found. Run `python -m scripts.run_sweep` first.")
    st.stop()

# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### Source")
    source = st.radio("CGM feed", ["Replay a recorded trace", "Nightscout (live)"],
                      label_visibility="collapsed")

    model_name = st.selectbox("Model", checkpoints,
                              index=checkpoints.index(best_checkpoint() or checkpoints[0]))
    fc = get_forecaster(model_name)

    nightscout_url = token = None
    patient_id = replay_at = None

    if source.startswith("Nightscout"):
        nightscout_url = st.text_input(
            "Nightscout URL", placeholder="my-cgm.up.railway.app",
            help="Your own instance. Most allow public read access; if yours "
                 "does not, add a read token below.",
        )
        token = st.text_input("Read token (optional)", type="password") or None
        st.caption(
            "Readings are fetched directly from the address you enter and are "
            "not stored or sent anywhere else."
        )
    else:
        patients = load_splits()["test"]
        patient_id = st.selectbox("Patient", patients)
        frame = cached_forecast(patient_id, fc)
        times = pd.to_datetime(frame["target_time"])
        # Default to a moment shortly before this patient's worst low, so the
        # demo opens on something worth watching.
        lows = frame["actual"] < HYPO_THRESHOLD
        default = (times[lows].iloc[len(times[lows]) // 2]
                   if lows.any() else times.iloc[len(times) // 2])
        replay_at = st.slider(
            "Replay clock", min_value=times.min().to_pydatetime(),
            max_value=times.max().to_pydatetime(),
            value=default.to_pydatetime(), format="YYYY-MM-DD HH:mm",
            help="Where 'now' sits in this recorded trace.",
        )

    st.markdown("---")
    st.markdown("### Alarm")
    target_fa = st.slider("Target false alarms per day", 1.0, 12.0, 6.0, 0.5)
    warmup_days = st.select_slider("Calibration warm-up", [7.0, 14.0, 28.0], value=14.0,
                                   format_func=lambda d: f"{d:g} days")

    st.markdown("---")
    st.markdown("### Phone alert")
    topic = st.text_input(
        "ntfy.sh topic", placeholder="glucoguard-<something-random>",
        help="Install the ntfy app, subscribe to this topic, and alerts arrive "
             "as push notifications. Anyone who knows the topic can read it, so "
             "pick something unguessable.",
    )
    dry_run = st.checkbox("Dry run (do not actually send)", value=True)

# --------------------------------------------------------------------------- #
# header
# --------------------------------------------------------------------------- #
ui.page(
    "Live",
    f"Runs the {HORIZON_MINUTES}-minute forecast against a CGM feed rather than a "
    "saved file, with the alarm threshold fitted to this wearer.",
    pills=["not a medical device", "no dose calculation"],
)

# --------------------------------------------------------------------------- #
# get the current window
# --------------------------------------------------------------------------- #
readings = None
if source.startswith("Nightscout"):
    if not nightscout_url:
        st.info("Enter a Nightscout address in the sidebar to begin.")
        st.stop()
    try:
        readings = fetch_nightscout(nightscout_url, count=64, token=token)
    except Exception as exc:                       # noqa: BLE001 — shown to user
        st.error(f"Could not read from Nightscout: {exc}")
        st.stop()
    window = to_window(readings)
    threshold_info = {"threshold": None,
                      "reason": "no calibration history for this feed yet"}
else:
    series = patient_series(patient_id)
    window = replay_window(series, as_utc(replay_at))
    readings = series[(pd.to_datetime(series["datetime"], utc=True)
                       <= as_utc(replay_at))].tail(48)
    threshold_info = personal_threshold(patient_id, model_name, target_fa, warmup_days)

# --------------------------------------------------------------------------- #
# refuse loudly when the feed is not good enough
# --------------------------------------------------------------------------- #
if window.values is None:
    st.markdown(
        f'<div class="gg-banner gg-caution"><span class="gg-icon">■</span>'
        f"<span><b>No forecast.</b> {window.reason}. The model needs "
        f"{HISTORY_STEPS * SAMPLE_MINUTES} unbroken minutes of recent readings, "
        f"and it is better to say nothing than to bridge a hole and present the "
        f"result as a prediction.</span></div>",
        unsafe_allow_html=True,
    )
    st.stop()

out = fc.predict_full(window.values[None, :])
mu = float(out["mu"][0])
sigma = float(out["sigma"][0]) if out["sigma"] is not None else None
risk = float(out["hypo_prob"][0]) if out["hypo_prob"] is not None else None
now_value = float(window.values[-1])

threshold = threshold_info.get("threshold")
using_personal = threshold is not None
if not using_personal:
    alarm_path = ARTIFACTS_DIR / "alarm.json"
    if alarm_path.exists() and risk is not None:
        entry = json.loads(alarm_path.read_text()).get(model_name, {})
        budgets = entry.get("budgets", {})
        chosen = budgets.get("6/day") or next(iter(budgets.values()), None)
        threshold = float(chosen["threshold"]) if chosen else 0.5

alarming = risk is not None and threshold is not None and risk >= threshold

# --------------------------------------------------------------------------- #
# the banner
# --------------------------------------------------------------------------- #
clock = window.last_time.strftime("%Y-%m-%d %H:%M UTC")
if alarming:
    st.markdown(
        f'<div class="gg-banner gg-alert"><span class="gg-icon">▲</span>'
        f"<span><b>Low glucose predicted.</b> Now {now_value:.0f} mg/dL, forecast "
        f"<b>{mu:.0f} mg/dL</b> in {HORIZON_MINUTES} minutes — "
        f"<b>{risk:.0%}</b> chance of being under {HYPO_THRESHOLD}. "
        f"Reading at {clock}.</span></div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="gg-banner gg-ok"><span class="gg-icon">●</span>'
        f"<span><b>No low predicted.</b> Now {now_value:.0f} mg/dL, forecast "
        f"{mu:.0f} mg/dL in {HORIZON_MINUTES} minutes"
        + (f" ({risk:.0%} chance of going under {HYPO_THRESHOLD})" if risk is not None else "")
        + f". Reading at {clock}.</span></div>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# chart: the window that produced the forecast, and where it points
# --------------------------------------------------------------------------- #
hist_times = pd.date_range(
    end=window.last_time, periods=HISTORY_STEPS, freq=f"{SAMPLE_MINUTES}min"
)
target_time = window.last_time + pd.Timedelta(minutes=HORIZON_MINUTES)

fig = go.Figure()
fig.add_hrect(y0=HYPO_THRESHOLD, y1=HYPER_THRESHOLD,
              fillcolor="rgba(12,163,12,0.055)", line_width=0, layer="below")
for value, colour, text in [(HYPO_THRESHOLD, CRITICAL, f"{HYPO_THRESHOLD} — low"),
                            (HYPER_THRESHOLD, WARNING, f"{HYPER_THRESHOLD} — high")]:
    fig.add_hline(y=value, line=dict(color=colour, width=1, dash="dot"),
                  annotation_text=text, annotation_position="right",
                  annotation_font=dict(size=11, color=INK_MUTED), layer="below")

fig.add_trace(go.Scatter(
    x=hist_times, y=window.values, name="CGM (last 2 hours)",
    mode="lines+markers", line=dict(color=ACTUAL, width=2),
    marker=dict(size=5), hovertemplate="%{y:.0f} mg/dL<extra></extra>",
))
if sigma is not None:
    fig.add_trace(go.Scatter(
        x=[target_time, target_time], y=[mu - 1.96 * sigma, mu + 1.96 * sigma],
        mode="lines", line=dict(color=PREDICTED, width=8),
        opacity=0.22, name="95% interval", hoverinfo="skip",
    ))
fig.add_trace(go.Scatter(
    x=[window.last_time, target_time], y=[now_value, mu],
    name=f"Forecast (+{HORIZON_MINUTES} min)", mode="lines+markers",
    line=dict(color=PREDICTED, width=2, dash="dash"),
    marker=dict(size=11, symbol="diamond"),
    hovertemplate="%{y:.0f} mg/dL<extra></extra>",
))

style(fig, height=380, y_title="mg/dL")
low = min(window.values.min(), mu - (1.96 * sigma if sigma else 0))
high = max(window.values.max(), mu + (1.96 * sigma if sigma else 0))
fig.update_yaxes(range=[max(20, low - 20), high + 25])
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# --------------------------------------------------------------------------- #
# calibration status and the alert button
# --------------------------------------------------------------------------- #
left, right = st.columns([3, 2])

with left:
    st.markdown('<div class="gg-h2">Alarm threshold</div>', unsafe_allow_html=True)
    if using_personal:
        st.markdown(
            f'<div class="gg-caption">Fitted to <b>this wearer</b> from their first '
            f"{warmup_days:g} days ({threshold_info['lows']:,} lows in that period), "
            f"targeting {target_fa:g} false alarms a day. Alarm fires at a predicted "
            f"risk of <b>{threshold:.0%}</b>; right now it is "
            f"<b>{risk:.0%}</b>.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="gg-caption">Using the shared population threshold '
            f"(<b>{threshold:.0%}</b>) because {threshold_info.get('reason', 'this feed has no history yet')}. "
            f"A shared threshold delivers a different alarm rate for every person — "
            f"see <code>CALIBRATION.md</code> — so a real deployment would switch to "
            f"a personal one after a couple of weeks of wear.</div>",
            unsafe_allow_html=True,
        )

with right:
    st.markdown('<div class="gg-h2">Send to phone</div>', unsafe_allow_html=True)
    if st.button("Send alert now", disabled=not topic, use_container_width=True):
        status = send_alert(topic, now_value, risk or 0.0, mu, dry_run=dry_run)
        (st.success if status.startswith("sent") or "dry run" in status
         else st.error)(status)
    st.markdown(
        '<div class="gg-caption">Sends one notification to the ntfy.sh topic you '
        "chose. Nothing is sent automatically and no data leaves this machine "
        "otherwise.</div>",
        unsafe_allow_html=True,
    )

ui.disclaimer()
