"""GlucoGuard, as a wearer would use it: connect, calibrate, then monitor.

The rest of this app is evidence for a reviewer. This page is the product, and
it is deliberately three steps long, because those three steps are the whole
thing a person has to do: point it at their sensor, let it learn what a normal
week looks like for them, and then leave it alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import product_ui as pu
from src import ui
from src.alarm_policy import tune_event_threshold
from src.calibration import split_by_time
from src.config import HISTORY_STEPS, HORIZON_MINUTES, HYPO_THRESHOLD, SAMPLE_MINUTES
from src.live import fetch_nightscout, send_alert, to_window
from src.monitor import (
    AlertRecord,
    MonitorState,
    SNOOZE_MINUTES,
    alert_text,
    should_alert,
    summarise,
    utc_now,
)
from src.predictor import (
    Forecaster,
    best_checkpoint,
    cached_forecast,
    load_splits,
    wearer_facts,
)

STEPS = ["1 · Connect", "2 · Calibrate", "3 · Monitor"]


@st.cache_resource(show_spinner=False)
def get_forecaster(name: str) -> Forecaster:
    return Forecaster(name)


@st.cache_data(show_spinner="Reading this wearer's history…")
def get_history(patient_id: str, model_name: str) -> pd.DataFrame:
    return cached_forecast(patient_id, get_forecaster(model_name))


@st.cache_data(show_spinner=False)
def fit_threshold(patient_id: str, model_name: str, target: float,
                  warmup_days: float) -> dict:
    """Fit this wearer's alarm cutoff on their own first weeks."""
    frame = get_history(patient_id, model_name)
    if "hypo_prob" not in frame.columns or frame.empty:
        return {"threshold": None, "reason": "this model has no risk output"}

    warm = split_by_time(frame["target_time"].to_numpy(), warmup_days)
    lows = int((frame.loc[warm, "actual"] < HYPO_THRESHOLD).sum())
    if lows < 20:
        return {"threshold": None, "lows": lows,
                "reason": f"only {lows} lows in the first {warmup_days:g} days"}

    thr = tune_event_threshold(frame.loc[warm, "actual"].to_numpy(),
                               frame.loc[warm, "hypo_prob"].to_numpy(), target)
    return {"threshold": float(thr), "lows": lows,
            "days": float(warmup_days), "windows": int(warm.sum())}


model_name = best_checkpoint()
if model_name is None:
    ui.page("GlucoGuard")
    st.error("No trained model found. Run `python -m scripts.run_sweep` first.")
    st.stop()
fc = get_forecaster(model_name)

state = st.session_state
# The step lives in the radio's own key. Passing `index=` and then writing the
# result back to a separate variable fights the widget: Streamlit remembers the
# widget's selection across reruns and ignores `index` after the first render,
# so the two drift apart and a click on step 2 can land on step 3.
state.setdefault("step_label", STEPS[0])
state.setdefault("source", "demo")
state.setdefault("patient", load_splits()["test"][0])
state.setdefault("target_fa", 6.0)
state.setdefault("warmup", 14.0)
state.setdefault("topic", "")
state.setdefault("monitor", MonitorState())
state.setdefault("running", False)
state.setdefault("cursor", None)

ui.page(
    "GlucoGuard",
    "Sees a low coming half an hour out, and lets you decide how often it is "
    "allowed to be wrong.",
    pills=["research demo", "not a medical device"],
)

st.radio("Step", STEPS, horizontal=True, key="step_label",
         label_visibility="collapsed")
step_index = STEPS.index(state.step_label)


def go_to(index: int) -> None:
    state.step_label = STEPS[index]
    st.rerun()

# --------------------------------------------------------------------------- #
# 1 · Connect
# --------------------------------------------------------------------------- #
if step_index == 0:
    ui.h2("Where should it read your glucose from?")
    choice = st.radio(
        "Source",
        ["Try it with a real recorded wearer", "Connect my own Nightscout"],
        label_visibility="collapsed",
    )

    if choice.startswith("Try"):
        state.source = "demo"
        patients = load_splits()["test"]
        state.patient = st.selectbox("Wearer", patients,
                                     index=patients.index(state.patient))
        facts = wearer_facts(state.patient)
        ui.tiles([
            ("Record shown", f"{facts['days']:,.0f} days", "of continuous wear"),
            ("Readings", f"{facts['readings']:,}", "every 5 minutes"),
            ("Time below 70", f"{facts['time_below_70']:.1%}",
             "clinical target is under 4%"),
            ("Seen in training", "No", "held-out wearer"),
        ])
        ui.caption(
            "These are donated traces from the OpenAPS Data Commons. This wearer "
            "was held out of training entirely, so what follows is the model "
            "meeting them for the first time."
        )
    else:
        state.source = "nightscout"
        url = st.text_input("Nightscout address",
                            placeholder="my-cgm.up.railway.app", key="ns_url")
        token = st.text_input("Read token (only if your site needs one)",
                              type="password", key="ns_token") or None
        if url:
            try:
                readings = fetch_nightscout(url, count=64, token=token)
                window = to_window(readings)
                if window.values is None:
                    ui.banner("caution", "Connected, but no usable window.",
                              f"{window.reason}.")
                else:
                    ui.banner("ok", "Connected.",
                              f"Latest reading {window.values[-1]:.0f} mg/dL at "
                              f"{window.last_time.strftime('%H:%M')} UTC.")
            except Exception as exc:                    # noqa: BLE001 — shown to user
                ui.banner("caution", "Could not read from that address.", str(exc))
        ui.caption(
            "Readings are fetched directly from the address you enter and are not "
            "stored or sent anywhere else."
        )

    if st.button("Next — calibrate", type="primary"):
        go_to(1)

# --------------------------------------------------------------------------- #
# 2 · Calibrate
# --------------------------------------------------------------------------- #
elif step_index == 1:
    ui.h2("How often may it interrupt you?")
    st.markdown(
        '<div class="gg-lead">Catching more lows always costs more false alarms. '
        "No setting avoids that trade, so you pick the side of it you can live "
        "with and GlucoGuard works out the cutoff that delivers it <i>for you</i> "
        "— from your own history, not from an average of other people.</div>",
        unsafe_allow_html=True,
    )

    state.target_fa = st.slider("False alarms a day you would accept",
                                1.0, 12.0, float(state.target_fa), 0.5)
    state.warmup = st.select_slider(
        "History to learn from", [7.0, 14.0, 28.0], value=float(state.warmup),
        format_func=lambda d: f"first {d:g} days",
    )

    if state.source == "demo":
        result = fit_threshold(state.patient, model_name,
                               state.target_fa, state.warmup)
        thr = result.get("threshold")
        if thr is not None:
            ui.banner("ok", "Calibrated.",
                      f"Your alarm fires when the chance of going low passes "
                      f"<b>{thr:.0%}</b>, learned from {result['lows']:,} low "
                      f"readings in your first {result['days']:.0f} days.")
            frame = get_history(state.patient, model_name)
            warm = split_by_time(frame["target_time"].to_numpy(), state.warmup)
            after = frame[~warm]
            fired = (after["hypo_prob"] >= thr).to_numpy()
            low = (after["actual"] < HYPO_THRESHOLD).to_numpy()
            days = len(after) * SAMPLE_MINUTES / (60 * 24)
            ui.tiles([
                ("Your cutoff", f"{thr:.0%}", "chance of going low"),
                ("Lows it catches", f"{(fired & low).sum() / max(low.sum(), 1):.0%}",
                 "of readings under 70, on the rest of your record"),
                ("False alarms", f"{(fired & ~low).sum() / days:.1f}",
                 f"a day, against the {state.target_fa:g} you asked for"),
                ("Learned from", f"{result['windows']:,}", "readings you already had"),
            ])
            ui.caption(
                "Everything above was measured on the part of the record the "
                "cutoff was <i>not</i> fitted on."
            )
        else:
            ui.banner("caution", "Not enough history to personalise yet.",
                      f"{result['reason']}. It falls back to a shared setting "
                      f"until you have worn it longer — and that shared setting "
                      f"is wrong for most people, which is why this step exists.")
    else:
        ui.banner("caution", "Personal calibration needs history.",
                  "A live feed has no past to learn from on day one. After a "
                  "fortnight this step fits your own cutoff; until then it uses "
                  "a shared one.")

    left, right = st.columns(2)
    if left.button("Back"):
        go_to(0)
    if right.button("Next — monitor", type="primary"):
        go_to(2)

# --------------------------------------------------------------------------- #
# 3 · Monitor
# --------------------------------------------------------------------------- #
else:
    if state.source != "demo":
        ui.banner("caution", "Live monitoring is on the Live page.",
                  "This step replays a recorded wearer, sped up, so a low "
                  "actually arrives while you are watching.")
        state.source = "demo"

    frame = get_history(state.patient, model_name)
    if frame.empty:
        st.warning("This wearer has no window with a complete 2-hour history.")
        st.stop()

    result = fit_threshold(state.patient, model_name, state.target_fa, state.warmup)
    threshold = result.get("threshold")
    times = pd.to_datetime(frame["target_time"])
    lows = (frame["actual"] < HYPO_THRESHOLD).to_numpy()

    if state.cursor is None:
        if lows.any():
            onsets = np.flatnonzero(lows & ~np.r_[False, lows[:-1]])
            state.cursor = int(max(0, onsets[len(onsets) // 2] - 40))
        else:
            state.cursor = len(frame) // 2

    controls, phone_col = st.columns([1, 1.3], gap="large")

    with controls:
        run_col, reset_col = st.columns(2)
        if run_col.button("Stop" if state.running else "Start monitoring",
                          type="primary", use_container_width=True):
            state.running = not state.running
            st.rerun()
        if reset_col.button("Reset", use_container_width=True):
            state.monitor = MonitorState()
            state.cursor = None
            state.running = False
            st.rerun()

        speed = st.select_slider(
            "Playback", [1, 6, 12, 24],
            value=state.get("speed", 12),
            format_func=lambda x: f"{x}x",
            help="How fast the recorded day plays. 12x means one CGM reading "
                 "every 25 seconds instead of every five minutes.",
            key="speed",
        )

        state.topic = st.text_input(
            "Phone alerts — ntfy.sh topic", value=state.topic,
            placeholder="glucoguard-something-random",
            help="Install ntfy, subscribe to this topic, and alerts arrive as "
                 "push notifications. Anyone who knows the topic can read it.",
        )
        live_send = st.toggle("Actually send", value=False,
                              help="Off by default. When off, alerts are logged "
                                   "but nothing leaves this machine.")

        st.markdown("---")
        # `empty` replaces its contents each tick; `container` would append,
        # so the counters would stack into a growing column of stale tiles.
        stats_slot = st.empty()

    # ----------------------------------------------------------------- #
    # one tick of the loop
    # ----------------------------------------------------------------- #
    def tick(advance: bool) -> None:
        if advance and state.cursor < len(frame) - 1:
            state.cursor += 1
            state.monitor.ticks += 1

        row = frame.iloc[state.cursor]
        # The clock that matters is the trace's, not the wall's — see should_alert.
        sim_now = times.iloc[state.cursor].to_pydatetime()
        now_g = float(row["current"])
        predicted = float(row["predicted"])
        risk = float(row["hypo_prob"]) if "hypo_prob" in frame.columns else None
        alarming = (risk is not None and threshold is not None
                    and risk >= threshold)

        if advance:
            fire, kind = should_alert(state.monitor, alarming, sim_now, now_g)
            if fire:
                delivered = send_alert(state.topic, now_g, risk or 0.0, predicted,
                                       dry_run=not (live_send and state.topic))
                state.monitor.log.append(AlertRecord(
                    at=sim_now, glucose=now_g, predicted=predicted,
                    risk=risk or 0.0, delivered=delivered, kind=kind,
                ))
                state.monitor.last_alert_at = sim_now
                state.monitor.last_state = kind
            elif not alarming and state.monitor.last_state != "ok":
                state.monitor.last_state = "ok"

        return row, now_g, predicted, risk, alarming

    @st.fragment(run_every=(5.0 / speed) if state.running else None)
    def live_panel() -> None:
        row, now_g, predicted, risk, alarming = tick(advance=state.running)
        picked = times.iloc[state.cursor]
        recent = frame["current"].iloc[
            max(0, state.cursor - HISTORY_STEPS + 1) : state.cursor + 1
        ].to_numpy(dtype=float)
        delta = float(recent[-1] - recent[-2]) if len(recent) > 1 else 0.0

        sim_now = picked.to_pydatetime()
        snoozed = (state.monitor.last_alert_at is not None
                   and alarming
                   and (sim_now - state.monitor.last_alert_at).total_seconds()
                   < SNOOZE_MINUTES * 60)

        st.markdown(
            pu.phone(now=now_g, predicted=predicted, risk=risk,
                     threshold=threshold, minutes=HORIZON_MINUTES,
                     delta_per_5min=delta, clock=picked.strftime("%H:%M"),
                     spark=list(recent), notification=alarming and not snoozed),
            unsafe_allow_html=True,
        )
        st.caption(
            f"{picked.strftime('%a %d %b %Y, %H:%M')} UTC · "
            + ("running" if state.running else "paused")
            + (f" · quiet for another "
               f"{SNOOZE_MINUTES - (sim_now - state.monitor.last_alert_at).total_seconds() / 60:.0f} min"
               if snoozed else "")
        )

        # These live inside the fragment on purpose. Rendered outside it they
        # only repaint on a full rerun, so the counters sit at zero while the
        # loop is plainly running — which reads as the app being broken.
        stats = summarise(state.monitor.log)
        with stats_slot.container():
            ui.tiles([
                ("Alerts sent", f"{stats['warnings']}", "in this session"),
                ("Readings seen", f"{state.monitor.ticks}", "since you pressed start"),
            ])

        with log_slot.container():
            ui.h2("Alert log")
            if state.monitor.log:
                st.dataframe(
                    pd.DataFrame([{
                        "When": a.at.strftime("%H:%M:%S"),
                        "Kind": a.kind,
                        "Glucose": f"{a.glucose:.0f}",
                        "Predicted": f"{a.predicted:.0f}",
                        "Risk": f"{a.risk:.0%}",
                        "Delivery": a.delivered,
                    } for a in reversed(state.monitor.log)]),
                    use_container_width=True, hide_index=True,
                )
                ui.caption(
                    f"One alert, then silence for {SNOOZE_MINUTES} minutes even if "
                    "glucose stays low — a low that lasts an hour is one event, not "
                    "twelve. It speaks again after an hour, because at that point "
                    "silence is indistinguishable from the app having crashed."
                )
            else:
                ui.caption(
                    "Nothing yet. Press **Start monitoring** and let it run — the "
                    "playback opens shortly before a real low, so you should not "
                    "have to wait long."
                )

    log_slot = st.empty()
    with phone_col:
        live_panel()

    # ----------------------------------------------------------------- #
    if st.button("Back to calibration"):
        go_to(1)

ui.disclaimer()
