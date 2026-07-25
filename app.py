"""GlucoGuard — 30-minute glucose forecasting demo.

Run:  streamlit run app.py

Everything on screen comes from patients in the held-out TEST split. The model
has never seen any of them during training, so what you are watching is the
model meeting a new person.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    ARTIFACTS_DIR,
    HISTORY_STEPS,
    HORIZON_MINUTES,
    HYPER_THRESHOLD,
    HYPO_THRESHOLD,
    SAMPLE_MINUTES,
)
from src.metrics import evaluate
from src.models.baselines import persistence
from src.predictor import (
    Forecaster,
    alarm_flags,
    available_checkpoints,
    best_checkpoint,
    hypo_episodes,
    hypo_lead_time,
    alarm_budgets,
    cached_forecast,
    load_splits,
    patient_series,
    tuned_threshold,
)
from src.theme import (
    ACTUAL,
    AXIS,
    BORDER,
    CRITICAL,
    CSS,
    GRID,
    INK_MUTED,
    PREDICTED,
    WARNING,
    style,
)

st.set_page_config(page_title="GlucoGuard", page_icon="🩸", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_forecaster(name: str) -> Forecaster:
    return Forecaster(name)


@st.cache_data(show_spinner=False)
def get_series(patient_id: str) -> pd.DataFrame:
    return patient_series(patient_id)


@st.cache_data(show_spinner="Running the model over this patient's record...")
def get_forecast(patient_id: str, model_name: str) -> pd.DataFrame:
    return cached_forecast(patient_id, get_forecaster(model_name))


@st.cache_data(show_spinner=False)
def get_sweep() -> dict | None:
    path = ARTIFACTS_DIR / "sweep.json"
    return json.loads(path.read_text()) if path.exists() else None


def tile(label: str, value: str, note: str = "") -> str:
    note_html = f'<div class="gg-tile-note">{note}</div>' if note else ""
    return (
        f'<div class="gg-tile"><div class="gg-tile-label">{label}</div>'
        f'<div class="gg-tile-value">{value}</div>{note_html}</div>'
    )


# --------------------------------------------------------------------------- #
# guard: the app needs a trained model
# --------------------------------------------------------------------------- #
checkpoints = available_checkpoints()
if not checkpoints:
    st.markdown('<div class="gg-title">GlucoGuard</div>', unsafe_allow_html=True)
    st.error(
        "No trained model found. Build the dataset and run the sweep first:\n\n"
        "```\npython -m src.data.build_dataset\npython -m scripts.run_sweep\n```"
    )
    st.stop()

# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
splits = load_splits()
test_patients = splits["test"]

with st.sidebar:
    st.markdown("### Controls")
    default_model = best_checkpoint() or checkpoints[0]
    model_name = st.selectbox(
        "Model", checkpoints, index=checkpoints.index(default_model),
        help="Every checkpoint the sweep produced. The default is the one that "
             "won on the validation split.",
    )
    patient_id = st.selectbox(
        "Patient (held-out test split)", test_patients,
        help="These patients were never used for training or model selection.",
    )

    series = get_series(patient_id)
    forecast = get_forecast(patient_id, model_name)

    if forecast.empty:
        st.warning("This patient has no window with a complete 2-hour history.")
        st.stop()

    days = pd.to_datetime(pd.Series(forecast["target_time"])).dt.date
    day_options = sorted(days.unique())

    # Default to the day holding the most low-glucose readings — the interesting one.
    lows_per_day = (
        pd.DataFrame({"day": days, "low": forecast["actual"].to_numpy() < HYPO_THRESHOLD})
        .groupby("day")["low"].sum()
    )
    default_day = int(day_options.index(lows_per_day.idxmax())) if lows_per_day.max() > 0 else 0

    day = st.selectbox(
        "Day", day_options, index=default_day,
        format_func=lambda d: d.strftime("%a %d %b %Y"),
        help="Defaults to this patient's day with the most time below 70 mg/dL.",
    )
    show_persistence = st.checkbox(
        "Overlay persistence baseline", value=False,
        help="'Assume glucose does not change.' The floor any forecaster must beat.",
    )

    budgets = alarm_budgets()
    budget = st.select_slider(
        "False alarms you would tolerate", options=budgets,
        value=budgets[len(budgets) // 2] if budgets else None,
        help="The alarm cutoff is tuned on validation patients to hit this "
             "budget, then applied unchanged here. Catching more lows always "
             "costs more false alarms — this is that dial.",
    ) if budgets else None

    st.markdown("---")
    fc = get_forecaster(model_name)
    st.caption(
        f"**{fc.arch.upper()}** · {fc.n_params:,} parameters\n\n"
        f"Input: {HISTORY_STEPS * SAMPLE_MINUTES} min of CGM "
        f"({HISTORY_STEPS} samples)\n\n"
        f"Output: glucose {HORIZON_MINUTES} min ahead"
    )

# --------------------------------------------------------------------------- #
# header
# --------------------------------------------------------------------------- #
st.markdown('<div class="gg-title">GlucoGuard</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="gg-sub">Forecasts blood glucose {HORIZON_MINUTES} minutes ahead from '
    f"continuous glucose monitor history, so a low can be seen coming instead of "
    f"reacted to. Every patient below is from the held-out test split — the model "
    f"has never seen their data.</div>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# day slice
# --------------------------------------------------------------------------- #
mask = days.to_numpy() == day
day_fc = forecast[mask].reset_index(drop=True)
target_time = pd.to_datetime(day_fc["target_time"])

# --------------------------------------------------------------------------- #
# alert banner — what the model is saying about this day
# --------------------------------------------------------------------------- #
threshold = tuned_threshold(fc, budget) if budget else None
predicted_low = pd.Series(alarm_flags(day_fc, threshold), index=day_fc.index)
actual_low = day_fc["actual"] < HYPO_THRESHOLD
caught = int((predicted_low & actual_low).sum())
missed = int((~predicted_low & actual_low).sum())
false_alarms = int((predicted_low & ~actual_low).sum())

if actual_low.any():
    first_low = target_time[actual_low].iloc[0].strftime("%H:%M")
    if caught:
        st.markdown(
            f'<div class="gg-banner gg-alert"><span class="gg-icon">▲</span>'
            f"<span><b>Low glucose predicted.</b> This patient went below "
            f"{HYPO_THRESHOLD} mg/dL on {len(day_fc[actual_low])} readings starting "
            f"{first_low}. The model flagged {caught} of them "
            f"{HORIZON_MINUTES} minutes in advance and missed {missed}.</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="gg-banner gg-caution"><span class="gg-icon">■</span>'
            f"<span><b>Low glucose occurred but was not predicted.</b> "
            f"{len(day_fc[actual_low])} readings fell below {HYPO_THRESHOLD} mg/dL "
            f"from {first_low} and the model called none of them. Days like this "
            f"are why the low-glucose metrics below are reported separately.</span></div>",
            unsafe_allow_html=True,
        )
elif false_alarms:
    st.markdown(
        f'<div class="gg-banner gg-caution"><span class="gg-icon">■</span>'
        f"<span><b>{false_alarms} false alarms.</b> Glucose stayed above "
        f"{HYPO_THRESHOLD} mg/dL all day, but the model predicted a low this many "
        f"times. Alarm fatigue is a real cost, so false alarms are tracked as a "
        f"headline metric rather than hidden.</span></div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="gg-banner gg-ok"><span class="gg-icon">●</span>'
        f"<span><b>No lows, no false alarms.</b> Glucose stayed above "
        f"{HYPO_THRESHOLD} mg/dL for the whole day and the model agreed "
        f"throughout.</span></div>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# main chart
# --------------------------------------------------------------------------- #
fig = go.Figure()

# Clinical target range as background, not as a data series.
fig.add_hrect(
    y0=HYPO_THRESHOLD, y1=HYPER_THRESHOLD,
    fillcolor="rgba(12,163,12,0.055)", line_width=0, layer="below",
)
for value, colour, text in [
    (HYPO_THRESHOLD, CRITICAL, f"{HYPO_THRESHOLD} — low"),
    (HYPER_THRESHOLD, WARNING, f"{HYPER_THRESHOLD} — high"),
]:
    fig.add_hline(
        y=value, line=dict(color=colour, width=1, dash="dot"),
        annotation_text=text, annotation_position="right",
        annotation_font=dict(size=11, color=INK_MUTED), layer="below",
    )

# Where the model reports its own spread, draw it — a forecast that admits
# doubt is more useful to a controller than a confident single number.
if "sigma" in day_fc.columns:
    fig.add_trace(go.Scatter(
        x=pd.concat([target_time, target_time[::-1]]),
        y=pd.concat([day_fc["predicted"] + 1.96 * day_fc["sigma"],
                     (day_fc["predicted"] - 1.96 * day_fc["sigma"])[::-1]]),
        fill="toself", fillcolor="rgba(235,104,52,0.09)",
        line=dict(width=0), hoverinfo="skip",
        name="95% predictive interval",
    ))

fig.add_trace(go.Scatter(
    x=target_time, y=day_fc["actual"], name="Actual CGM",
    mode="lines", line=dict(color=ACTUAL, width=2),
    hovertemplate="Actual %{y:.0f} mg/dL<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=target_time, y=day_fc["predicted"],
    name=f"Predicted {HORIZON_MINUTES} min earlier",
    mode="lines", line=dict(color=PREDICTED, width=2, dash="dash"),
    hovertemplate="Predicted %{y:.0f} mg/dL<extra></extra>",
))

if show_persistence:
    fig.add_trace(go.Scatter(
        x=target_time, y=day_fc["current"], name="Persistence baseline",
        mode="lines", line=dict(color=AXIS, width=1.5),
        hovertemplate="Persistence %{y:.0f} mg/dL<extra></extra>",
    ))

# Mark the moments the model called a low and was right — the point of the system.
hit = predicted_low & actual_low
if hit.any():
    fig.add_trace(go.Scatter(
        x=target_time[hit], y=day_fc.loc[hit, "actual"],
        name="Low correctly predicted", mode="markers",
        marker=dict(color=CRITICAL, size=9, line=dict(color="#fcfcfb", width=2)),
        hovertemplate="Low called %d min early<extra></extra>" % HORIZON_MINUTES,
    ))

style(fig, height=430, y_title="mg/dL")
# Keep the axis on the two lines. The predictive interval is deliberately wide
# and letting it drive the range would squash the signal it is drawn around.
lo = day_fc[["actual", "predicted"]].min().min()
hi = day_fc[["actual", "predicted"]].max().max()
fig.update_yaxes(range=[max(20, lo - 30), hi + 35])
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
st.markdown(
    f'<div class="gg-caption">Both lines are drawn at the time they describe. '
    f"The dashed line was produced {HORIZON_MINUTES} minutes before that moment, "
    f"using only data available then — so the gap between the lines is the error a "
    f"patient would actually have experienced.</div>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# risk strip — only for models that emit one
# --------------------------------------------------------------------------- #
if "hypo_prob" in day_fc.columns:
    risk = go.Figure()
    risk.add_trace(go.Scatter(
        x=target_time, y=day_fc["hypo_prob"] * 100,
        mode="lines", line=dict(color=CRITICAL, width=2),
        fill="tozeroy", fillcolor="rgba(208,59,59,0.12)",
        name="Predicted risk of going low",
        hovertemplate="%{y:.0f}% risk of being under 70<extra></extra>",
    ))
    if threshold is not None:
        risk.add_hline(
            y=threshold * 100, line=dict(color=CRITICAL, width=1, dash="dot"),
            annotation_text=f"alarm at {threshold:.0%}", annotation_position="right",
            annotation_font=dict(size=11, color=INK_MUTED),
        )
    style(risk, height=170, y_title="P(low) %")
    risk.update_yaxes(range=[0, 100])
    risk.update_layout(showlegend=False)
    st.plotly_chart(risk, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        f'<div class="gg-caption">This model outputs a probability, not just a '
        f"number, which is what makes the alarm a tunable decision. The dotted "
        f"line is the cutoff tuned on <i>validation</i> patients to stay within "
        f"<b>{budget}</b> false alarms — move the slider and watch it trade "
        f"missed lows against false alarms. A fixed 70 mg/dL rule has no such "
        f"dial.</div>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# metrics for this patient (whole record, not just the day on screen)
# --------------------------------------------------------------------------- #
m_model = evaluate(forecast["actual"].to_numpy(), forecast["predicted"].to_numpy())
m_persist = evaluate(forecast["actual"].to_numpy(), forecast["current"].to_numpy())

# Alarm statistics must follow the tuned cutoff, not the 70 mg/dL one baked into
# `evaluate` — otherwise the tiles would contradict the slider above them.
record_alarm = alarm_flags(forecast, threshold)
record_low = (forecast["actual"] < HYPO_THRESHOLD).to_numpy()
tp = float((record_alarm & record_low).sum())
fp = float((record_alarm & ~record_low).sum())
alarm_precision = tp / (tp + fp) if tp + fp else 0.0
fa_per_day = fp / (len(forecast) * SAMPLE_MINUTES / (60 * 24))
median_lead, caught_share = hypo_lead_time(forecast, threshold)
episodes = hypo_episodes(forecast, threshold)
improvement = (m_persist["rmse"] - m_model["rmse"]) / m_persist["rmse"] * 100

st.markdown(f'<div class="gg-h2">Patient {patient_id} — full record</div>',
            unsafe_allow_html=True)
cols = st.columns(5)
cols[0].markdown(
    tile("RMSE", f"{m_model['rmse']:.1f} <span style='font-size:.9rem;font-weight:400'>mg/dL</span>",
         f"{improvement:+.0f}% vs persistence ({m_persist['rmse']:.1f})"),
    unsafe_allow_html=True)
cols[1].markdown(
    tile("Low episodes caught", f"{caught_share:.0%}",
         f"of {len(episodes):,} separate lows, warned before onset"),
    unsafe_allow_html=True)
cols[2].markdown(
    tile("Median warning",
         f"{median_lead:.0f} <span style='font-size:.9rem;font-weight:400'>min</span>"
         if median_lead else "—",
         "ahead of glucose crossing 70"),
    unsafe_allow_html=True)
cols[3].markdown(
    tile("Alarm precision", f"{alarm_precision:.0%}",
         f"{fa_per_day:.1f} false alarms per day at this setting"),
    unsafe_allow_html=True)
cols[4].markdown(
    tile("Clarke A+B", f"{m_model['clarke_ab']:.1f}%",
         "clinically acceptable predictions"),
    unsafe_allow_html=True)
st.markdown(
    '<div class="gg-caption">An <b>episode</b> is one continuous stretch below '
    "70 mg/dL. Counting episodes rather than readings avoids letting a single "
    "long low inflate the score, and matches what a patient actually "
    "experiences: one event that either was or was not seen coming.</div>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# where the model actually wins
# --------------------------------------------------------------------------- #
st.markdown('<div class="gg-h2">Where the accuracy comes from</div>',
            unsafe_allow_html=True)

bands = [
    ("Below 70", forecast["actual"] < HYPO_THRESHOLD),
    ("70–180", (forecast["actual"] >= HYPO_THRESHOLD) & (forecast["actual"] <= HYPER_THRESHOLD)),
    ("Above 180", forecast["actual"] > HYPER_THRESHOLD),
]
labels, model_rmse, persist_rmse = [], [], []
for label, sel in bands:
    if sel.sum() < 30:
        continue
    labels.append(label)
    err_m = forecast.loc[sel, "predicted"] - forecast.loc[sel, "actual"]
    err_p = forecast.loc[sel, "current"] - forecast.loc[sel, "actual"]
    model_rmse.append(float(np.sqrt((err_m**2).mean())))
    persist_rmse.append(float(np.sqrt((err_p**2).mean())))

bar = go.Figure()
bar.add_trace(go.Bar(
    x=labels, y=persist_rmse, name="Persistence",
    marker=dict(color=AXIS, cornerradius=4),
    hovertemplate="Persistence %{y:.1f} mg/dL<extra></extra>",
))
bar.add_trace(go.Bar(
    x=labels, y=model_rmse, name=fc.arch.upper(),
    marker=dict(color=ACTUAL, cornerradius=4),
    hovertemplate=f"{fc.arch.upper()} %{{y:.1f}} mg/dL<extra></extra>",
))
bar.update_layout(barmode="group", bargap=0.35, bargroupgap=0.06)
style(bar, height=300, y_title="RMSE (mg/dL)", x_title="Actual glucose band")
bar.update_layout(hovermode="x")
st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})
st.markdown(
    '<div class="gg-caption">Split by what glucose actually did. The low band is '
    "the hardest and the most clinically important: it is rare, so a model can post "
    "a strong overall RMSE while being weak exactly where a patient needs it.</div>",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# model comparison across the whole test split
# --------------------------------------------------------------------------- #
sweep = get_sweep()
if sweep:
    st.markdown('<div class="gg-h2">All models, full test split</div>',
                unsafe_allow_html=True)
    table = pd.DataFrame([
        {
            "Model": r["name"],
            "Params": f"{r.get('n_params', 0):,}",
            "RMSE": round(r["test"]["rmse"], 2),
            "MAE": round(r["test"]["mae"], 2),
            "RMSE (lows)": round(r["test"]["rmse_hypo"], 2),
            "Low recall": f"{r['test']['hypo_recall']:.1%}",
            "Precision": f"{r['test']['hypo_precision']:.1%}",
            "Clarke A+B": f"{r['test']['clarke_ab']:.1f}%",
        }
        for r in sweep["results"]
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.markdown(
        f'<div class="gg-caption">Selected on the validation split: '
        f'<b>{sweep["selected_on_validation"]}</b>. The test column was computed '
        f"once, after selection, and never used to choose anything.</div>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# honesty section
# --------------------------------------------------------------------------- #
with st.expander("What this does not do"):
    st.markdown(
        f"""
- **It does not recommend insulin.** The output is a glucose forecast and a
  low-glucose warning. Nothing here computes a dose, and treating it as if it
  did would be unsafe.
- **It has not been tested on a person.** Everything above is a retrospective
  replay of recorded CGM traces from the OpenAPS Data Commons. There has been no
  prospective trial and no in-silico closed-loop evaluation yet.
- **It uses CGM only.** Insulin and carbohydrate records exist in the dataset
  and are not used yet, which is why the model is weakest right after meals and
  corrections — the moments its inputs cannot see.
- **It reports one horizon.** {HORIZON_MINUTES} minutes. Longer horizons are
  substantially harder and are not claimed here.
- **Uncertainty is not modelled.** The model returns a single number with no
  confidence attached, so it cannot yet say when it does not know — the next
  thing to build.
"""
    )
