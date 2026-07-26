"""Overview: the front page of the app."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src import results_io as rio
from src import ui
from src.config import HORIZON_MINUTES, HYPO_THRESHOLD, SAMPLE_MINUTES, HISTORY_STEPS
from src.theme import ACTUAL, AXIS, INK_MUTED, MUTED_LINE, PREDICTED, style

ui.page(
    "GlucoGuard",
    f"Blood glucose forecast {HORIZON_MINUTES} minutes ahead, turned into a "
    "low-glucose alarm that a wearer can actually tune. Trained on 28,281 "
    "patient-days of real CGM traces and evaluated on people the model has "
    "never seen.",
    pills=["research demo", "not a medical device"],
)

# --------------------------------------------------------------------------- #
# headline numbers
# --------------------------------------------------------------------------- #
sweep = rio.load("sweep")
matched = rio.load("matched")
external = rio.external_summary()
policy = rio.load("policy")

best_rmse = persistence_rmse = None
if sweep:
    by_name = {r["name"]: r for r in sweep["results"]}
    persistence_rmse = by_name.get("persistence", {}).get("test", {}).get("rmse")
    tcn_prob = rio._metrics_file("tcn_prob")
    best_rmse = (tcn_prob or by_name.get("tcn", {})).get("test", {}).get("rmse")

recall_15 = None
if matched:
    ranked = sorted(matched["test"].items(), key=lambda kv: -kv[1][str(matched["rates"][-1])])
    recall_15 = ranked[0][1][str(matched["rates"][-1])]

ext_rmse = external["regression"]["model"]["rmse"] if external else None
policy_recall = (policy["results"]["test"]["rolling"]["episode_recall"]
                 if policy else None)

ui.tiles([
    ("Forecast error",
     f"{best_rmse:.1f}" if best_rmse else ", ",
     f"mg/dL RMSE · persistence {persistence_rmse:.1f}" if persistence_rmse else ""),
    ("Low episodes warned",
     f"{policy_recall:.0%}" if policy_recall else ", ",
     "at 6 false alarms a day, per-wearer threshold"),
    ("On a second population",
     f"{ext_rmse:.1f}" if ext_rmse else ", ",
     "mg/dL, different sensors, no retraining"),
    ("Held-out wearers",
     f"{len(sweep['counts']['test']['patients']) if False else sweep['counts']['test']['patients']}"
     if sweep else ", ",
     f"{sweep['counts']['test']['windows']:,} scored windows" if sweep else ""),
])

# --------------------------------------------------------------------------- #
# what the thing does, in one picture
# --------------------------------------------------------------------------- #
ui.h2("What it does")
st.markdown(
    f'<div class="gg-lead">A continuous glucose monitor is a very good sensor '
    f"and a very late alarm: it tells you that you are low, which is already too "
    f"late, and at 3 a.m. it is competing with sleep. GlucoGuard reads the last "
    f"{HISTORY_STEPS * SAMPLE_MINUTES} minutes of CGM and predicts where glucose "
    f"will be {HORIZON_MINUTES} minutes from now, so a low can be seen coming."
    "</div>",
    unsafe_allow_html=True,
)

# A small illustrative trace so the front page is not all tables.
t = np.arange(0, 150, 5)
actual = 140 - 0.55 * t + 6 * np.sin(t / 11)
pred = actual + np.random.default_rng(3).normal(0, 4, len(t))
fig = go.Figure()
fig.add_hrect(y0=HYPO_THRESHOLD, y1=180, fillcolor="rgba(12,163,12,0.055)",
              line_width=0, layer="below")
fig.add_hline(y=HYPO_THRESHOLD, line=dict(color="#d03b3b", width=1, dash="dot"),
              annotation_text=f"{HYPO_THRESHOLD}, low", annotation_position="right",
              annotation_font=dict(size=11, color=INK_MUTED))
fig.add_trace(go.Scatter(x=t[:20], y=actual[:20], name="CGM so far",
                         mode="lines", line=dict(color=ACTUAL, width=2.4)))
fig.add_trace(go.Scatter(x=t[19:], y=pred[19:], name=f"Forecast (+{HORIZON_MINUTES} min)",
                         mode="lines", line=dict(color=PREDICTED, width=2.4, dash="dash")))
style(fig, height=250, y_title="mg/dL", x_title="minutes")
fig.update_layout(showlegend=True)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
ui.caption(
    "Illustrative. The real traces, and what the model actually said about them, "
    "are on the Patient explorer page."
)

# --------------------------------------------------------------------------- #
# the argument, in three claims
# --------------------------------------------------------------------------- #
ui.h2("The three findings this project rests on")

left, mid, right = st.columns(3)
with left:
    ui.h3("Accuracy fought safety")
    st.markdown(
        '<div class="gg-caption">Rank the models by RMSE and you almost exactly '
        "reverse their ranking by low-glucose recall. Squared error rewards a "
        "forecast that hugs the mean, and hypoglycaemia is the tail, so "
        "optimising accuracy taught the model to avoid committing to the events "
        "it exists to catch. <b>Selecting on RMSE would have shipped the worst "
        "available alarm.</b></div>",
        unsafe_allow_html=True,
    )
with mid:
    ui.h3("The alarm is a decision, not a number")
    st.markdown(
        '<div class="gg-caption">Reading every model at a fixed 70 mg/dL cutoff '
        "compares their biases, not their skill: the high-recall ones simply "
        "alarm more often. Tuning each to the same false-alarm budget inverts the "
        "ranking, linear extrapolation goes from apparently best to last.</div>",
        unsafe_allow_html=True,
    )
with right:
    ui.h3("The threshold belongs to the wearer")
    st.markdown(
        '<div class="gg-caption">One shared cutoff delivers wildly different '
        "alarm rates to different people, 3 to 26 a day across the external "
        "cohort. Fitting each wearer on their own first weeks puts almost all of "
        "them on the rate they asked for.</div>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------- #
# where to go
# --------------------------------------------------------------------------- #
ui.h2("Explore")
st.markdown(
    """
| Page | What is on it |
|---|---|
| **Patient explorer** | Replay a real day for a held-out wearer, with the alarm dial |
| **Models** | Every model trained, and why the most accurate one is not the one that ships |
| **Alarm** | Recall against false alarms, matched budgets, and the alarm policy |
| **Calibration** | Per-wearer thresholds, how they move over time, and what that costs |
| **Generalisation** | A second population with different sensors, and drift over years |
| **Inputs** | What happens when insulin and carbohydrate records are added |
| **Method** | Data, splits, and the rules that make these numbers mean something |
""",
    unsafe_allow_html=True,
)

ui.disclaimer()
