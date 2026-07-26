"""How the forecast becomes a warning, and how alarms should be compared."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src import results_io as rio
from src import ui
from src.config import HYPO_THRESHOLD
from src.theme import ACTUAL, INK_MUTED, INK_SECONDARY, MUTED_LINE, PREDICTED, style


ui.page(
    "Alarm",
    "A forecast is a number; an alarm is a decision. This page is about the gap "
    "between them, which turned out to matter more than any modelling choice.",
    pills=["held-out test split"],
)

ui.note(
    "The model outputs a probability that glucose will be under "
    f"{HYPO_THRESHOLD} mg/dL in half an hour. To alarm or not, that probability "
    "needs a cutoff — and the model does not supply one. The cutoff is a "
    "statement about how much interruption a person will tolerate, so it is "
    "chosen by asking for a <b>false-alarm budget</b> and finding the threshold "
    "that delivers it."
)

# --------------------------------------------------------------------------- #
ui.h2("Recall against false alarms")
alarm = rio.alarm_curves()
if alarm:
    default = [n for n in ("tcn_prob", "tcn", "ridge", "persistence",
                           "linear_extrapolation") if n in alarm]
    picked = st.multiselect("Models", sorted(alarm), default=default)
    fig = go.Figure()
    palette = {n: c for n, c in zip(picked, [ACTUAL, PREDICTED, "#1baf7a",
                                             INK_SECONDARY, MUTED_LINE,
                                             "#4a3aa7", "#e34948"])}
    for name in picked:
        c = alarm[name]["pr_curve_test"]
        pairs = [(f, r * 100) for f, r in zip(c["false_alarms_per_day"], c["recall"])
                 if f <= 24]
        if not pairs:
            continue
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pairs], y=[p[1] for p in pairs], name=name,
            mode="lines", line=dict(width=2.4, color=palette.get(name, MUTED_LINE)),
            hovertemplate=f"{name}<br>%{{x:.1f}} false alarms/day · "
                          "recall %{y:.1f}%<extra></extra>",
        ))
    style(fig, height=430, y_title="Low-glucose recall (%)",
          x_title="False alarms per day")
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(showgrid=True, gridcolor="#e1e0d9")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    ui.caption(
        "The threshold swept across its whole range. A single (recall, precision) "
        "pair is one arbitrary point on this curve, which is why quoting one "
        "compares thresholds rather than models."
    )

# --------------------------------------------------------------------------- #
ui.h2("Read at a matched false-alarm rate")
ui.table(rio.matched_table(), "Run `python -m scripts.matched_comparison`.")
ui.note(
    "Compare this with the fixed-cutoff numbers on the Models page. "
    "<b>Linear extrapolation reaches 74% recall at a fixed 70 mg/dL cutoff and "
    "comes last here</b> — its apparent advantage was 21 alarms a day, which "
    "nobody would wear. Matching the rate is the only comparison that is not "
    "decided by the threshold."
)

# --------------------------------------------------------------------------- #
ui.h2("Counting the way a wearer counts")
policy = rio.load("policy")
if policy:
    ui.note(
        "Per-reading accounting treats each five-minute sample as its own alarm "
        f"opportunity, so half an hour of nuisance alarming is six false alarms. "
        f"Here an alarm fires and stays quiet for {policy['refractory_minutes']} "
        "minutes, a low <b>episode</b> counts as warned if a sound was made in the "
        "hour before glucose crossed 70, and an alarm is false only if no low "
        "followed. Alarms during an ongoing low are not false."
    )
    import pandas as pd
    rows = []
    for cohort, res in policy["results"].items():
        for pol in ("shared", "fixed", "rolling"):
            m = res.get(pol)
            if not m:
                continue
            rows.append({
                "Cohort": cohort,
                "Threshold policy": {"shared": "one for everybody",
                                     "fixed": "per-wearer, fitted once",
                                     "rolling": "per-wearer, re-fitted weekly"}[pol],
                "Low episodes": f"{m['episodes']:,}",
                "Episodes warned": f"{m['episode_recall']:.1%}",
                "False alarms/day": round(m["false_alarms_per_day"], 1),
                "Total alarms/day": round(m["alarms_per_day"], 1),
                "Median warning": f"{m['median_lead_minutes']:.0f} min",
                "Wearers within 2x of target": f"{m['per_wearer_fa']['within_2x']:.0%}",
            })
    ui.table(pd.DataFrame(rows))
    ui.caption(
        "The shared threshold's higher recall is bought with nearly twice the "
        "interruptions, and it still misses the requested rate for a third of "
        "wearers."
    )
else:
    st.info("Run `python -m scripts.eval_policy` to populate this.")

ui.disclaimer()
