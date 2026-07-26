"""Data, splits, and the rules that make the rest of these numbers mean something."""
from __future__ import annotations

import streamlit as st

from src import results_io as rio
from src import ui
from src.config import (
    HISTORY_STEPS, HORIZON_MINUTES, HYPER_THRESHOLD, HYPO_THRESHOLD,
    MAX_INTERPOLATION_GAP_STEPS, SAMPLE_MINUTES, SEED,
)


ui.page(
    "Method",
    "Most of the engineering here went into not fooling ourselves. This page is "
    "the part that makes the other pages readable as evidence.",
)

# --------------------------------------------------------------------------- #
ui.h2("The task")
st.markdown(
    f"""
| | |
|---|---|
| Input | {HISTORY_STEPS} CGM samples — {HISTORY_STEPS * SAMPLE_MINUTES} minutes at {SAMPLE_MINUTES}-minute spacing |
| Output | glucose {HORIZON_MINUTES} minutes after the last sample, as a distribution |
| Decision | is this person heading below {HYPO_THRESHOLD} mg/dL |
| Not the task | insulin dosing, longer horizons, anything prospective |
"""
)

ui.h2("Data")
sweep = rio.load("sweep")
if sweep:
    c = sweep["counts"]
    ui.tiles([
        ("Train", f"{c['train']['patients']} wearers", f"{c['train']['windows']:,} windows"),
        ("Validation", f"{c['val']['patients']} wearers", f"{c['val']['windows']:,} windows"),
        ("Test", f"{c['test']['patients']} wearers", f"{c['test']['windows']:,} windows"),
        ("External", "33 wearers", "3.9M windows, never read in training"),
    ])
st.markdown(
    """
The [OpenAPS Data Commons](https://openaps.org/outcomes/data-commons/) is a
6.5 GB archive of Nightscout exports donated by people running open-source
automated insulin delivery. 40 wearers were selected by data volume, giving
**8,145,010 readings over 28,281 patient-days**. The raw archive is not
redistributed here.
"""
)

# --------------------------------------------------------------------------- #
ui.h2("The four rules")

ui.h3("1. Split by patient, never by row")
ui.caption(
    "CGM samples five minutes apart are enormously autocorrelated. Split rows at "
    "random and the model can memorise a trace it has already seen a few steps "
    "earlier, which produces a beautiful test score and a system that fails on the "
    f"first new person it meets. The assignment is fixed with seed {SEED} and "
    "written to <code>artifacts/splits.json</code>."
)

ui.h3("2. Baselines first")
ui.caption(
    f"At a {HORIZON_MINUTES}-minute horizon, glucose is autocorrelated enough that "
    "<i>persistence</i> — predicting no change at all — is already a decent "
    "forecaster. Any result that omits it can make a mediocre model look "
    "impressive. Linear extrapolation and ridge regression on the raw window are "
    "reported too, so every neural number reads as <i>how much did the extra "
    "complexity buy</i>."
)

ui.h3("3. Interpolate short gaps; discard windows that span long ones")
ui.caption(
    f"Gaps up to {MAX_INTERPOLATION_GAP_STEPS * SAMPLE_MINUTES} minutes are filled "
    "linearly. Anything longer invalidates every window that crosses it, because "
    "filling a three-hour hole with a straight line invents data and quietly "
    "inflates the score. A window's prediction target must be a genuinely observed "
    "reading, never an interpolated one."
)

ui.h3("4. Lows get their own metrics")
ui.caption(
    f"Readings below {HYPO_THRESHOLD} mg/dL are about 3% of the data, so a model "
    "can post an excellent overall RMSE while being useless exactly where a "
    "patient needs it. RMSE restricted to lows, alarm recall and precision, false "
    "alarms per day, and the Clarke Error Grid are all reported separately."
)

# --------------------------------------------------------------------------- #
ui.h2("Metrics, defined")
st.markdown(
    f"""
| Metric | Definition |
|---|---|
| RMSE / MAE | forecast error in mg/dL against the observed value {HORIZON_MINUTES} min later |
| RMSE (lows) | the same, restricted to windows whose true value is under {HYPO_THRESHOLD} |
| MARD | mean absolute relative difference, the CGM industry's accuracy convention |
| Clarke A+B | share of predictions in zones that would not lead to wrong treatment |
| Episode recall | share of continuous stretches below {HYPO_THRESHOLD} preceded by an alarm |
| False alarms/day | de-duplicated alarm events that no low followed, per day of wear |
| Time below 70 | share of a wearer's readings under {HYPO_THRESHOLD} — clinical target is under 4% |
"""
)
ui.note(
    "Ranges follow the ATTD/ADA international consensus on CGM metrics: target "
    f"{HYPO_THRESHOLD}–{HYPER_THRESHOLD} mg/dL, with under 4% of time below "
    f"{HYPO_THRESHOLD}."
)

# --------------------------------------------------------------------------- #
ui.h2("Mistakes we caught, and what they cost")
st.markdown(
    """
| What was wrong | How it was found | What changed |
|---|---|---|
| Alarm recall read at a fixed 70 mg/dL cutoff | Linear extrapolation "won" while alarming 21×/day | Every model now scored at matched false-alarm rates |
| The shipped model was chosen by reading test recall | Top three sat within 1.6 points | Selection moved to two folds inside validation |
| The external set shared four donors with train and test | Checked ids across both export formats | Those wearers excluded before any scoring |
| Lead time counted warnings issued *after* glucose crossed 70 | Median lead came out as exactly 0 | Rewritten to require the alert before onset |
| Headline tables labelled by false-alarm *budget*, not achieved rate | A literature check flagged the mismatch | All tables relabelled to achieved rates |
| Rolling recalibration "pulled ahead over time" | The external cohort reversed the trend | Claim withdrawn; the two policies are indistinguishable here |
"""
)
ui.caption(
    "This table is here because a results page without one is usually a results "
    "page that stopped looking."
)

ui.h2("Reproducing it")
st.code(
    "pip install -r requirements.txt\n"
    "python -m src.data.build_dataset      # raw archive -> tidy parquet\n"
    "python -m scripts.run_sweep           # baselines + architectures\n"
    "python -m scripts.run_round2          # probabilistic and classifier heads\n"
    "python -m scripts.select_model        # choose on validation only\n"
    "python -m scripts.eval_external       # a second population\n"
    "python -m scripts.eval_policy         # event-level alarm comparison\n"
    "streamlit run app.py",
    language="bash",
)

ui.disclaimer()
