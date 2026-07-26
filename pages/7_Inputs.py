"""What happens when the model is allowed to see insulin and carbohydrates."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src import results_io as rio
from src import ui
from src.theme import ACTUAL, MUTED_LINE, PREDICTED, style


ui.page(
    "Inputs",
    "The shipped model reads CGM and nothing else, so it is blind at exactly the "
    "moments a person acts. The archive holds what they did and what their loop "
    "computed. These four experiments measure how much of that blindness each "
    "one fills.",
    pills=["same architecture", "same split", "same budget"],
)

ui.note(
    "Only the input channels change. Same TCN, same patient split, same 12 epochs "
    "at the same learning rate, same seed. Parameter counts differ by 0.3%, so "
    "<b>this compares information rather than capacity</b>."
)

mm = rio.load("multimodal")
if not mm:
    st.info(
        "The multimodal sweep has not finished. Run "
        "`python -m src.data.build_multimodal` then `python -m scripts.run_multimodal`."
    )
    ui.h2("What the archive actually contains")
else:
    ui.h2("Results")
    ui.table(rio.multimodal_table())

    rows = mm["results"]
    base = next((r["test"]["rmse"] for r in rows if r["name"] == "cgm"), None)
    labels = {"cgm": "CGM only", "treatments": "+ what the wearer did",
              "devicestatus": "+ what the loop computed", "both": "+ both"}
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[labels.get(r["name"], r["name"]) for r in rows],
        y=[r["test"]["rmse"] for r in rows],
        marker=dict(color=[MUTED_LINE if r["name"] == "cgm" else ACTUAL for r in rows],
                    cornerradius=4),
        hovertemplate="%{x}<br>test RMSE %{y:.2f} mg/dL<extra></extra>",
    ))
    if base:
        fig.add_hline(y=base, line=dict(color=MUTED_LINE, width=1, dash="dot"))
    style(fig, height=330, y_title="test RMSE (mg/dL)")
    fig.update_layout(showlegend=False, hovermode="x")
    lo = min(r["test"]["rmse"] for r in rows)
    hi = max(r["test"]["rmse"] for r in rows)
    fig.update_yaxes(range=[lo - 0.6, hi + 0.3])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    ui.caption("Dotted line is the CGM-only baseline. Lower is better.")

# --------------------------------------------------------------------------- #
ui.h2("What the archive actually contains")
st.markdown(
    """
| Channel | Source | Present | Non-zero | What it is |
|---|---|---:|---:|---|
| bolus | `treatments.json` | 100% | 5.1% | insulin units the wearer delivered |
| carbohydrates | `treatments.json` | 100% | 1.2% | grams entered at a meal |
| basal rate | `treatments.json` | 51% | 27% | temporary basal the pump was running |
| insulin on board | `devicestatus.json` | 20% | 20% | the loop's own estimate of active insulin |
| carbs on board | `devicestatus.json` | 20% | 5.7% | the loop's estimate of unabsorbed carbs |
| insulin activity | `devicestatus.json` | 20% | 20% | rate of insulin action, from the loop's model |
"""
)
ui.note(
    "Two very different kinds of information. <b>Treatments are what happened</b> — "
    "hand-entered, so under-reported, but ground truth about actions. "
    "<b>devicestatus is what OpenAPS calculated</b>, already convolved through its "
    "pharmacokinetic model, which saves the network rediscovering insulin action "
    "curves but is only present a fifth of the time."
)

ui.h3("Absent action and absent state are not the same thing")
ui.caption(
    "A bolus that was not recorded probably did not happen, so zero is the right "
    "fill. Insulin-on-board is a running estimate that is simply missing when the "
    "loop was not reporting — filling it with zero would assert <i>no insulin "
    "active</i>, which is a different and usually false claim. So state channels "
    "carry a per-source <b>observedness flag</b>, and the two sources get separate "
    "flags because their coverage differs by a factor of three. Without that, the "
    "network could not tell a genuine zero from a silent gap."
)

ui.disclaimer()
