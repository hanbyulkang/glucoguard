"""Every model trained, and why the most accurate one is not the one that ships."""
from __future__ import annotations

import json

import plotly.graph_objects as go
import streamlit as st

from src import results_io as rio
from src import ui
from src.config import ARTIFACTS_DIR
from src.theme import ACTUAL, INK_MUTED, MUTED_LINE, PREDICTED, style


ui.page(
    "Models",
    "Eleven models, all scored on the same eight held-out wearers. The table is "
    "sorted by accuracy, read it, then read the chart below it, which shows why "
    "sorting by accuracy is the wrong thing to do.",
    pills=["held-out test split"],
)

ui.h2("Every model, on held-out wearers")
df = rio.model_table()
ui.table(df, "Run `python -m scripts.run_sweep` to populate this.")
ui.caption(
    "RMSE and MAE are in mg/dL. <b>RMSE (lows)</b> is restricted to windows whose "
    "true value is under 70, the region that matters clinically and the one an "
    "overall average hides. <b>Clarke A+B</b> is the share of predictions in the "
    "clinically acceptable zones of the Clarke Error Grid."
)

# --------------------------------------------------------------------------- #
sweep = rio.load("sweep")
if sweep:
    ui.h2("Accuracy and sensitivity to lows pull in opposite directions")
    rows = [r for r in sweep["results"] if not r["name"].startswith("ensemble")]
    x = [r["test"]["rmse"] for r in rows]
    y = [r["test"]["hypo_recall"] * 100 for r in rows]
    names = [r["name"] for r in rows]
    learned = [not n.startswith(("persistence", "linear", "ridge")) for n in names]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers+text", text=names, textposition="top center",
        textfont=dict(size=11, color=INK_MUTED),
        marker=dict(size=14, color=[ACTUAL if L else MUTED_LINE for L in learned],
                    line=dict(color="#fcfcfb", width=2)),
        hovertemplate="%{text}<br>RMSE %{x:.2f} · recall %{y:.1f}%<extra></extra>",
    ))
    style(fig, height=430, y_title="Low-glucose recall (%)",
          x_title="RMSE (mg/dL), lower is better")
    fig.update_layout(showlegend=False, hovermode="closest")
    fig.update_xaxes(showgrid=True, gridcolor="#e1e0d9")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    ui.note(
        "Each dot is a model, read at a fixed 70 mg/dL alarm cutoff. <b>The best "
        "model on the x-axis is the worst on the y-axis.</b> This is what squared "
        "error does: it rewards a forecast that stays near the conditional mean, "
        "and hypoglycaemia lives in the tail, so the accurate model becomes the "
        "reluctant one. It is also why a fixed cutoff is the wrong way to compare "
        "alarms, see the Alarm page."
    )

# --------------------------------------------------------------------------- #
# Per-epoch histories are a build artefact and are not carried in the deploy
# bundle. Announcing their absence under a heading reads as a broken page, so
# the whole section stands down when there is nothing to draw.
curves = {}
for stem in ("tcn", "transformer", "lstm", "tcn_hypo3", "tcn_prob", "tcn_cls", "tcn_mt"):
    blob = rio._metrics_file(stem)
    if blob and blob.get("history"):
        curves[stem] = blob["history"]

if curves:
    ui.h2("Training curves")
    picked = st.multiselect("Show", sorted(curves), default=["tcn", "transformer", "lstm"])
    fig = go.Figure()
    palette = [ACTUAL, PREDICTED, "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948"]
    for colour, name in zip(palette, picked):
        h = curves[name]
        fig.add_trace(go.Scatter(
            x=[e["epoch"] for e in h], y=[e["rmse"] for e in h], name=name,
            mode="lines+markers", line=dict(width=2.2, color=colour),
            marker=dict(size=5),
            hovertemplate=f"{name}<br>epoch %{{x}} · val RMSE %{{y:.2f}}<extra></extra>",
        ))
    style(fig, height=360, y_title="validation RMSE (mg/dL)", x_title="epoch")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    ui.caption(
        "Validation RMSE per epoch. Model selection used this curve; the test "
        "split was scored once afterwards and never used to choose anything."
    )


# --------------------------------------------------------------------------- #
ui.h2("Which one ships")
sel = rio.load("selection")
if sel:
    st.markdown(
        f'<div class="gg-lead">The shipped model is <b>{sel["selected"]}</b>. '
        "It was chosen on validation wearers only, split into two folds, a "
        "threshold tuned on one and scored on the other, because the top three "
        "sit within 1.6 points of each other and picking among near-ties by "
        "reading the test set turns noise into a decision.</div>",
        unsafe_allow_html=True,
    )
    rows = []
    for name, by_budget in sel["val_recall"].items():
        row = {"Model": name}
        row.update({k: f"{v:.1%}" for k, v in by_budget.items()})
        rows.append(row)
    import pandas as pd
    frame = pd.DataFrame(rows)
    frame["mean"] = [
        f"{sum(sel['val_recall'][n].values()) / len(sel['val_recall'][n]):.1%}"
        for n in frame["Model"]
    ]
    ui.table(frame.sort_values("mean", ascending=False).reset_index(drop=True))
    ui.caption("Validation-only recall at each false-alarm budget. Test is not involved.")
else:
    st.info("Run `python -m scripts.select_model` to populate this.")

ui.disclaimer()
