# Devpost project page copy

Paste into the matching fields. All numbers are the measured ones.

---

## Project name

**GlucoGuard**

## Elevator pitch (≤200 characters)

> Continuous glucose monitors tell you that you're low. GlucoGuard tells you
> you're about to be — forecasting blood glucose 30 minutes ahead from 28,000
> patient-days of real CGM data.

*(198 characters)*

## Built with

`python` `pytorch` `pandas` `numpy` `scikit-learn` `streamlit` `plotly` `pyarrow`

---

## Inspiration

A continuous glucose monitor is a very good sensor and a very late alarm. It
measures where blood sugar is right now, so when it warns you about
hypoglycaemia, you are already hypoglycaemic. At 3 a.m. that alarm is also
competing with sleep, and severe nocturnal lows are one of the outcomes people
with type 1 diabetes fear most.

The gap is not sensing. It is anticipation. We wanted to know how much warning
you can actually extract from the CGM signal alone — no new hardware, no extra
inputs, just the trace the device is already producing.

## What it does

GlucoGuard reads the last two hours of CGM history and predicts blood glucose 30
minutes into the future. That forecast drives one decision: **is this person
heading below 70 mg/dL?**

The demo replays real recorded days from patients the model has never seen. You
watch the actual glucose trace and the model's forecast side by side — where
every forecast point was produced half an hour before the moment it describes —
plus the moments the model called a low before it started, and how much warning
it gave.

It does not recommend insulin doses. It is a forecast and a warning.

## How we built it

**Data.** The OpenAPS Data Commons is a 6.5 GB archive of Nightscout exports
donated by people running open-source automated insulin delivery. We read the
CGM records straight out of the zip, kept sensor glucose only, clipped to a
plausible 20–450 mg/dL, and resampled every patient onto a regular 5-minute grid.
That gives **40 patients, 8.1 million readings, 28,281 patient-days**.

**The split is by patient, never by row.** This is the decision the whole project
rests on. CGM samples five minutes apart are enormously autocorrelated; split
rows at random and the model can memorise a trace it has already seen a few steps
earlier, which produces a beautiful test score and a system that fails on the
first new person it meets. 26 patients train, 6 select, 8 are held out entirely.

**Models.** Persistence and linear extrapolation as non-learned floors, ridge
regression on the raw window, then an LSTM, a dilated TCN, and a Transformer
encoder — all predicting the *change* in glucose rather than the level, and all
given the rate of change as an explicit second input channel. Plus an ensemble
and a loss-reweighting variant that pushes the model to care more about lows.

**Metrics.** Overall RMSE is not enough. Readings below 70 mg/dL are about 3% of
the data, so a model can look excellent on average while being useless exactly
where a patient needs it. We report RMSE restricted to lows, the recall and
precision of the low-glucose alarm, false alarms per day, and the Clarke Error
Grid — the standard measure of whether a glucose error would lead to the wrong
treatment.

## Challenges we ran into

**The archive fights you.** 6.5 GB, two incompatible export formats, one patient
holding 1.5 GB by themselves, and CGM traces riddled with dropouts. We stream the
relevant files out of the zip rather than extracting it, and exclude that one
outlier patient so a single person cannot dominate training.

**Deciding what a gap means.** Filling a three-hour hole with a straight line
invents data and quietly inflates the score. We interpolate gaps up to 15 minutes
and discard any window that spans a longer one — and a window's prediction target
must be a genuinely observed reading, never an interpolated one.

**Our first lead-time metric was wrong.** It counted an episode as "caught" when
the warning was actually issued *after* glucose had already crossed 70 — a
warning that arrives late is not a warning. Rewriting it to require the alert
before onset cut the reported catch rate substantially, which is the honest
number.

**Stacking every fix at once made things worse.** The variant with a predicted
distribution, a classification head and a reweighted loss all together scored
below the one that just predicts a distribution. We reported that rather than
quietly dropping it.

**The two splits are not equally hard,** and we say so. Our validation patients
spend far less time low than our test patients do. That is real between-person
variation, and it means validation RMSE sits well below test RMSE for every
model alike. Selection still works — ranking is what selection needs — but we
refuse to quote the validation number as performance.

## Accomplishments that we're proud of

**We caught our own evaluation lying to us, twice, and the fix became the
project.**

First: rank our models by RMSE and you almost exactly reverse their ranking by
low-glucose recall. Our most accurate model was the worst at catching lows —
worse than doing nothing at all. That is what squared error does. It rewards a
forecast that hugs the mean, and hypoglycaemia is the tail, so optimising
accuracy taught the model to refuse to commit to the exact events we built it
for. Had we selected on RMSE, as almost every paper in this area does, we would
have shipped the worst available alarm.

Second: the models that looked good on recall weren't better, they just alarmed
more often. Linear extrapolation reaches 74% recall by firing 21 times a day.
Reading every model at one fixed 70 mg/dL cutoff compares their biases, not
their skill.

So the alarm became a tunable decision instead of a side effect. Every model
emits a risk score; the cutoff is tuned on validation to a false-alarm budget and
applied unchanged to test. Compared that way the ranking inverts — linear
extrapolation goes from apparently best to last — and the model that ships
catches **74.8% of lows against the plain network's 69.9% at identical accuracy
and an identical false-alarm budget**. Same architecture, same 18.9 mg/dL RMSE.
Only the decision layer changed.

## What we learned

That the hard part of medical machine learning is the measurement, not the model.
Every meaningful decision we made was about what number to trust — and most of
our debugging time went into finding places where our own evaluation was being
too kind to us.

## What's next for GlucoGuard

1. **Insulin and carbohydrate inputs.** The archive already contains them. The
   model is currently blind right after meals and corrections, which is exactly
   where it is weakest.
2. **Per-patient alarm calibration.** A threshold tuned on one group of people
   overshoots its false-alarm budget on another, so a wearer's own first weeks
   should set their cutoff.
3. **Withhold predictions on bad input.** Sensor dropouts and out-of-distribution
   traces should produce silence, not a confident wrong number. The predictive
   spread is the hook this hangs on and it already exists.
4. **A 60-minute horizon**, reported separately rather than quoting the easy one.

Longer term, this forecast is the sensing-and-prediction layer of a wearable that
integrates glucose sensing and insulin delivery in one housing. That work is
outside this submission and involves no human insulin delivery.

---

## Note on prior work (include this — the rules require care here)

The background research for this project — reading the CGM forecasting
literature and obtaining the OpenAPS Data Commons dataset — was done before the
hackathon, which the rules explicitly encourage. **All code in this submission
was written during the hackathon period and the repository's commit history
reflects that.** No pre-existing codebase was submitted.
