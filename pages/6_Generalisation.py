"""Does any of this survive contact with people the model has never seen?"""
from __future__ import annotations

import streamlit as st

from src import results_io as rio
from src import ui


ui.page(
    "Generalisation",
    "Held-out wearers answer whether this works on a new person. They do not "
    "answer whether it works on a different kind of person. This page is about "
    "the second question, and about what happens over years rather than weeks.",
    pills=["external population", "no retraining"],
)

# --------------------------------------------------------------------------- #
ui.h2("A second population, different hardware")
ext = rio.external_summary()
if ext:
    m, p = ext["regression"]["model"], ext["regression"]["persistence"]
    ui.tiles([
        ("Wearers", f"{ext['patients']}", f"{ext['windows']:,} scored windows"),
        ("Time below 70", f"{ext['hypo_rate']:.2%}",
         "test wearers were at 4.29%"),
        ("RMSE", f"{m['rmse']:.2f}", f"persistence {p['rmse']:.2f} mg/dL"),
        ("Clarke A+B", f"{m['clarke_ab']:.1f}%", "clinically acceptable"),
    ])
    ui.note(
        "These are AndroidAPS exports from the untouched half of the archive — a "
        "different app, a sensor mix including Medtronic and Abbott Libre, mostly "
        "European wearers. None of it was read when the training data was "
        "assembled. <b>Four donors had uploaded under both export formats — one a "
        "test wearer, three training wearers — and are excluded</b>; without that "
        "check this would have been a re-test on people the model already knew. "
        "The model was applied unchanged, thresholds and all."
    )
    st.markdown(
        """
| | original test | external population |
|---|---:|---:|
| RMSE | 18.86 | 19.95 |
| RMSE vs persistence | −19% | −16% |
| Clarke A+B | 96.3% | 97.3% |
| recall at 8 false alarms/day | 59.5% | 63.4% |
| recall at 15 false alarms/day | 75.3% | 77.9% |
| persistence at 15 | 64.3% | 65.3% |
"""
    )
    ui.note(
        "<b>The ranking survives and the calibration does not.</b> At every matched "
        "false-alarm rate the model still beats persistence, by 9 to 13 points, and "
        "the external numbers are if anything slightly better. But a threshold tuned "
        "for six false alarms a day delivers 14.7 on the test wearers and 10.5 here."
    )
else:
    st.info("Run `python -m scripts.eval_external` to populate this.")

# --------------------------------------------------------------------------- #
ui.h2("Does the calibration go stale?")
ui.table(rio.drift_table(), "Run `python -m scripts.eval_drift`.")
ui.caption(
    "A threshold fitted on a fortnight and then left alone. Six false alarms a "
    "day requested; 6.5 delivered in weeks 3–4 and 6.2 past the two-year mark. "
    "The external cohort's slow decline tracks those wearers going low less often "
    "over the same period — an alarm firing less because the person needs it "
    "less, not a threshold decaying."
)

# --------------------------------------------------------------------------- #
ui.h2("Does anything improve with wear?")
ui.table(rio.over_time_table(), "Run `python -m scripts.eval_over_time`.")
ui.note(
    "No, and it should not: the network is frozen after training and never sees "
    "another gradient. Episode recall holds at 74–79% from month one past year "
    "two. <b>Holding steady is the useful outcome</b> — a wearer who calibrated "
    "two years ago still gets the alarm they asked for — but it is not the system "
    "learning."
)

ui.h2("Was the rising RMSE real?")
ui.table(rio.within_patient_table(), "Run `python -m scripts.eval_within_patient`.")
ui.caption(
    "Pooled RMSE rose from 17.1 to 21.0 across the buckets, which looks like a "
    "model going stale. Comparing each wearer against their own average, the "
    "within-wearer slope is +0.07 mg/dL per bucket on test and +0.19 externally — "
    "essentially flat. <b>The pooled rise was composition</b>: the wearer count "
    "falls from 7 to 4 and 25 to 9, so later rows describe different people."
)

ui.disclaimer()
