# Devpost submission — Vitalitics 2026

Paste into the matching fields. Every number here is measured; nothing is
rounded up in our favour.

**Deadline: 2026-07-30 21:00 PDT** — the Devpost value is 2026-07-31 00:00 EDT,
so July 31 never actually happens.

---

## Project name

**GlucoGuard**

## Elevator pitch (≤200 characters)

> A continuous glucose monitor tells you that you're low. GlucoGuard tells you
> you're about to be — and lets you set how often it's allowed to be wrong.

*(139 characters)*

## Built with

`python` `pytorch` `pandas` `numpy` `scikit-learn` `scipy` `streamlit` `plotly`
`pyarrow`

---

## Inspiration

A continuous glucose monitor is a very good sensor and a very late alarm. It
measures where blood sugar is right now, so when it warns about hypoglycaemia
you are already hypoglycaemic. At 3 a.m. that alarm is also competing with
sleep, and severe nocturnal lows are among the outcomes people with type 1
diabetes fear most.

The gap is not sensing. It is anticipation. We wanted to know how much warning
you can extract from the CGM signal alone — no new hardware, no extra inputs,
just the trace the device is already producing.

## What it does

GlucoGuard reads the last two hours of CGM and predicts a **distribution** over
blood glucose 30 minutes ahead. That distribution drives one decision: *what is
the probability this person goes below 70 mg/dL?*

On eight held-out wearers it warns about **77% of low-glucose episodes before
they begin**, at six false alarms a day, with a median of 25 minutes of warning.

The alarm threshold is not a constant. It is fitted to each wearer from their
own first two weeks, because a single population-wide cutoff delivers wildly
different alarm rates to different people — 3 to 26 a day across our external
cohort.

It does not recommend insulin doses.

## How we built it

**Data.** The OpenAPS Data Commons: a 6.5 GB archive of Nightscout exports
donated by people running open-source automated insulin delivery. We read CGM
records straight out of the zip, kept sensor glucose only, clipped to a
plausible 20–450 mg/dL, and resampled onto a regular 5-minute grid. That gives
**40 wearers, 8.1 million readings, 28,281 patient-days**.

**The split is by patient, never by row.** This is the decision the whole
project rests on. CGM samples five minutes apart are enormously autocorrelated;
split rows at random and the model memorises a trace it saw a few steps earlier,
producing a beautiful test score and a system that fails on the first new person
it meets. 26 wearers train, 6 select, 8 are held out entirely.

**Models.** Persistence and linear extrapolation as non-learned floors, ridge
regression on the raw window, then LSTM, dilated TCN, and Transformer — all
predicting the *change* in glucose rather than the level, with rate of change as
an explicit second channel. Then a probabilistic head (Gaussian NLL), a
classification head trained directly on the low/not-low label, and loss
re-weighting toward lows.

**The alarm is a separate layer.** Each model emits a risk score; the cutoff on
that score is tuned on validation to a false-alarm budget, then applied
unchanged to test. Alarms fire once and stay quiet for 30 minutes, and are
scored on **episodes** rather than readings, because that is what a wearer
experiences.

## Challenges we ran into

**We caught our own evaluation lying to us, four times.** Each one changed the
project more than any modelling choice did.

*Ranking by RMSE almost exactly reverses the ranking by low-glucose recall.*
Our most accurate model was the worst at catching lows — worse than doing
nothing. Squared error rewards a forecast that hugs the mean and hypoglycaemia
is the tail, so optimising accuracy taught the model to refuse to commit to the
exact events we built it for. Selecting on RMSE, as is standard in this area,
would have shipped the worst available alarm.

*Reading every model at a fixed 70 mg/dL cutoff compares their biases, not their
skill.* Linear extrapolation reached 74% recall by alarming 21 times a day.
Matched to the same false-alarm budget it comes **last**.

*Our first lead-time metric counted warnings issued after glucose had already
crossed 70.* The median came out as exactly zero, which is what exposed it.

*We were choosing the shipped model by reading test recall.* The top three sat
within 1.6 points of each other — exactly where selecting on test turns noise
into a decision. Selection moved inside validation, using two patient folds.

**And the archive fights you.** Two incompatible export formats, one wearer
holding 1.5 GB alone, traces riddled with dropouts. We stream the relevant files
out of the zip, interpolate gaps up to 15 minutes, and discard any window
spanning a longer one — filling a three-hour hole with a straight line invents
data and quietly inflates the score.

## Accomplishments that we're proud of

**The evaluation is the accomplishment**, and its best moment was catching a
contamination nobody would have found by accident. We built a second cohort from
the untouched half of the archive — 33 AndroidAPS wearers, different app,
different sensors, mostly European — and discovered that **four donors had
uploaded under both export formats**, one of them a test wearer. Without that
check, our "external validation" would have been a re-test on people the model
already knew.

Applied unchanged to that population, the model holds: RMSE 19.95 against
persistence's 23.77, Clarke A+B **97.3%**, and at every matched false-alarm rate
it still beats persistence by 9 to 13 points. **The ranking survives; the
calibration does not** — which is precisely the evidence that sent us to
per-wearer thresholds.

## What we learned

That the hard part of medical machine learning is the measurement, not the
model. Every meaningful decision came from finding a place where our own
evaluation was being too kind to us.

The clearest example arrived last. Adding insulin and carbohydrate records
improved validation RMSE, made **test** RMSE worse, and improved the alarm on
both. We concluded "the extra inputs failed" — from RMSE — and had to withdraw
that an hour later after scoring them as alarms. It is the exact mistake this
project exists to warn about, committed by us, on our own work.

## What's next for GlucoGuard

1. **Separate SMB from meal boluses.** One wearer logs 71 boluses a day at a
   median of 0.20 U; another logs 0.5 a day at 3.5 U. The first is a loop
   micro-dosing, the second is a person eating. The same number in the same
   channel means opposite things, and that is the likeliest reason treatment
   inputs transfer badly between cohorts.
2. **Withhold predictions on out-of-distribution input.** The predictive spread
   already exists; it should gate the output, not just decorate it.
3. **A prospective evaluation.** Everything here is retrospective replay.
4. **Longer horizons, reported separately** rather than quoting the easy one.

Longer term this forecast is the sensing-and-prediction layer of a wearable that
integrates glucose sensing and insulin delivery in one housing. That work is
outside this submission and involves no human insulin delivery.

---

## Note on prior work — include this, the rules are ambiguous here

Background research (reading the CGM forecasting literature, obtaining the
OpenAPS Data Commons dataset) was done before the hackathon, which the rules
explicitly encourage. **All code in this submission was written during the
hackathon period and the repository's commit history reflects that.** No
pre-existing codebase was submitted.
