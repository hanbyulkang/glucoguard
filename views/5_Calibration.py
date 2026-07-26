"""Per-wearer thresholds: why they are needed, what they cost, how they move."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src import results_io as rio
from src import ui
from src.theme import ACTUAL, INK_MUTED, MUTED_LINE, style


ui.page(
    "Calibration",
    "Two independent population shifts broke the alarm threshold and neither "
    "broke the model's ranking. That points at a design: let the model supply "
    "the ordering, and let each wearer's own history set their cutoff.",
    pills=["per-wearer", "strictly causal"],
)

# --------------------------------------------------------------------------- #
ui.h2("Every wearer needs a different cutoff")
thr = rio.threshold_table()
if thr is not None and not thr.empty:
    cohort = st.radio("Cohort", sorted(thr["Cohort"].unique()), horizontal=True)
    sub = thr[thr["Cohort"] == cohort].dropna(subset=["Threshold"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["Time below 70"], y=sub["Threshold"], mode="markers",
        marker=dict(size=13, color=ACTUAL, line=dict(color="#fcfcfb", width=2)),
        text=sub["Wearer"],
        hovertemplate="%{text}<br>threshold %{y:.1f}% · "
                      "time below 70: %{x:.2f}%<extra></extra>",
    ))
    style(fig, height=380, y_title="Personal alarm threshold, P(low) %",
          x_title="Share of this wearer's readings under 70 mg/dL")
    fig.update_layout(showlegend=False, hovermode="closest")
    fig.update_xaxes(showgrid=True, gridcolor="#e1e0d9")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    ui.note(
        "If a single rule worked, these dots would lie on a line. They do not: a "
        "wearer who is low 6.7% of the time gets a 4.6% cutoff, and one who is low "
        "1.2% of the time gets 16.2%. <b>How often someone goes low does not tell "
        "you how confident the model gets about them</b>, so the threshold has to "
        "be measured per person rather than derived."
    )
    ui.table(sub.drop(columns=["Cohort"]).reset_index(drop=True))
    missing = thr[(thr["Cohort"] == cohort) & (thr["Threshold"].isna())]
    if not missing.empty:
        ui.caption(
            f"<b>{len(missing)} wearers could not be calibrated</b> — their warm-up "
            "held fewer than 20 lows. They fall back to the shared cutoff, and they "
            "are exactly the people the shared cutoff serves worst."
        )
else:
    st.info("Run `python -m scripts._thr_fast` to populate this.")

# --------------------------------------------------------------------------- #
ui.h2("What per-wearer calibration buys, and what it costs")
ui.table(rio.calibration_table(), "Run `python -m scripts.eval_calibration`.")
ui.note(
    "The column that matters is the last one. A shared cutoff does not miss the "
    "target by a little — it misses by different amounts for different people. "
    "Per-wearer calibration costs roughly five points of pooled recall at the "
    "same achieved rate, and that price is worth understanding: <b>a single "
    "global threshold earns part of its pooled score by treating people "
    "unequally</b>, letting frequent-low wearers alarm constantly for cheap true "
    "positives while rare-low wearers get almost nothing."
)

# --------------------------------------------------------------------------- #
ui.h2("The threshold follows the wearer")
traj = rio.trajectory()
if traj:
    cohort = st.radio("Cohort ", sorted(traj), horizontal=True, key="traj_cohort")
    wearers = sorted(traj[cohort])
    who = st.selectbox("Wearer", wearers)
    r = traj[cohort][who]
    sched = r["schedule"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[p["days_since_start"] for p in sched],
        y=[p["trailing_hypo_rate"] * 100 for p in sched],
        name="recent time below 70", mode="lines",
        line=dict(color=MUTED_LINE, width=1.8, dash="dash"),
        hovertemplate="day %{x:.0f} · %{y:.1f}% of readings low<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[p["days_since_start"] for p in sched],
        y=[p["threshold"] * 100 for p in sched],
        name="alarm threshold", mode="lines",
        line=dict(color=ACTUAL, width=2.4),
        hovertemplate="day %{x:.0f} · threshold %{y:.1f}%<extra></extra>",
    ))
    style(fig, height=360, y_title="%", x_title="days worn")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    th = r["threshold"]
    ui.tiles([
        ("Correlation with own lows", f"{r['corr_threshold_vs_rate']:+.2f}",
         "threshold against recent time below 70"),
        ("Range", f"{th['min']:.1%} – {th['max']:.1%}", "over the whole record"),
        ("Weekly movement", f"{th['median_weekly_step']:.1%}",
         "median step, gradual not jumpy"),
        ("Refits", f"{r['refits']}", f"over {r['weeks']} weeks"),
    ])
    ui.caption(
        "Re-fitting weekly on the trailing month, the cutoff does not settle on a "
        "number and stay there. When someone starts going low more often the model "
        "hands out high probabilities more often, so the bar has to rise to keep "
        "interruptions at the requested rate."
    )
else:
    st.info("Run `python -m scripts.eval_trajectory` to populate this.")

ui.disclaimer()
