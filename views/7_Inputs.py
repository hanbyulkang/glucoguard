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

ui.banner(
    "caution", "Read the alarm table, not the RMSE table.",
    "Adding insulin and carbohydrate records made validation RMSE better and "
    "test RMSE worse, and at the same time made the low-glucose alarm better on "
    "both. We drew the wrong conclusion from RMSE first. The same disagreement "
    "between accuracy and sensitivity that this whole project is about applies "
    "to its own input experiment.",
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
ui.h2("The same models, scored as alarms")
mma = rio.load("multimodal_alarm")
if mma:
    import pandas as pd
    rows = []
    base = mma["results"].get("cgm", {}).get("matched", {})
    for name, r in mma["results"].items():
        row = {"Inputs": labels.get(name, name) if mm else name}
        for rate, v in r["matched"].items():
            delta = f" ({v - base[rate]:+.1%})" if base and name != "cgm" else ""
            row[f"{rate} FA/day"] = f"{v:.1%}{delta}"
        p = r["events"]["personal"]
        row["Episodes warned (per-wearer cutoff)"] = f"{p['episode_recall']:.1%}"
        row["FA/day"] = round(p["false_alarms_per_day"], 1)
        rows.append(row)
    ui.table(pd.DataFrame(rows))
    ui.note(
        "<b>Insulin and carbohydrate records do help the alarm</b>, 1 to 2 points "
        "of recall at every matched false-alarm rate, despite costing 0.44 mg/dL of "
        "RMSE. But look at the last two columns: <b>once each wearer has their own "
        "threshold the difference disappears</b> (77.4% against 77.6%). Per-wearer "
        "calibration and treatment inputs are two ways of solving the same problem, "
        "and doing one absorbs most of what the other would have given."
    )
    ui.caption(
        "The loop-derived channels contribute nothing and cost recall at the "
        "tightest budget. At 20% coverage there is not enough of them to learn from."
    )
else:
    st.info("Run `python -m scripts.eval_multimodal_alarm` to populate this.")

ui.h2("Why the treatment channels are harder than they look")
ui.note(
    "One wearer records 71 boluses a day at a median of 0.20 U; another records "
    "0.5 a day at a median of 3.5 U. That is not sloppy record-keeping, the first "
    "is running <b>SMB</b>, where the loop delivers a micro-bolus every few minutes "
    "instead of adjusting basal, and the second boluses only at meals. "
    "<b>The same number in the same channel means opposite things</b>, and the "
    "encoding here does not distinguish them. That, rather than the data being "
    "wrong, is the most likely reason these channels transfer badly between "
    "cohorts with different treatment habits."
)

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
    "Two very different kinds of information. <b>Treatments are what happened</b>, "
    "hand-entered, so under-reported, but ground truth about actions. "
    "<b>devicestatus is what OpenAPS calculated</b>, already convolved through its "
    "pharmacokinetic model, which saves the network rediscovering insulin action "
    "curves but is only present a fifth of the time."
)

ui.h3("Absent action and absent state are not the same thing")
ui.caption(
    "A bolus that was not recorded probably did not happen, so zero is the right "
    "fill. Insulin-on-board is a running estimate that is simply missing when the "
    "loop was not reporting, filling it with zero would assert <i>no insulin "
    "active</i>, which is a different and usually false claim. So state channels "
    "carry a per-source <b>observedness flag</b>, and the two sources get separate "
    "flags because their coverage differs by a factor of three. Without that, the "
    "network could not tell a genuine zero from a silent gap."
)

ui.disclaimer()
