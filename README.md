# GlucoGuard

**Predicting low blood sugar 30 minutes before it happens.**

A continuous glucose monitor tells someone with type 1 diabetes what their blood
sugar is *now*. By the time it alarms, the low has already started — and if it
starts at 3 a.m., the alarm is competing with sleep. GlucoGuard reads the last
two hours of CGM history and forecasts where glucose will be half an hour from
now, so a low can be seen coming instead of reacted to.

Trained on 40 people and **28,281 patient-days** of real CGM data from the
[OpenAPS Data Commons](https://openaps.org/outcomes/data-commons/), and
evaluated on eight people the model never saw during training.

```bash
pip install -r requirements.txt
python -m src.data.build_dataset     # raw archive -> tidy parquet
python -m scripts.run_sweep          # baselines + 3 architectures + ensemble
streamlit run app.py                 # the demo
```

---

## What it does

Given 24 CGM samples (two hours, five minutes apart), predict the glucose value
30 minutes after the last one. The output feeds a single clinical decision:
**is this person heading below 70 mg/dL?**

It does **not** recommend insulin doses. See [Limitations](#limitations).

---

## Results

Full numbers are in [`results.md`](results.md); the alarm comparison is in
[`alarm.md`](alarm.md).

A dilated convolutional network cuts RMSE from persistence's 23.2 mg/dL to
18.9 on held-out patients. But the headline result is the one we did not expect:

![Accuracy and low-glucose sensitivity move in opposite directions](assets/tradeoff.png)

**Rank the models by accuracy and you almost exactly reverse their ranking by
low-glucose recall.** The most accurate model is the least willing to call a low.

This is what squared error does. It rewards a forecast that stays near the
conditional mean, and hypoglycaemia lives in the tail — so a model that hedges
toward the middle wins on RMSE precisely by refusing to commit to the events the
product exists to catch. Selecting on RMSE would have shipped the worst
available alarm.

Two things follow. First, recall at a fixed 70 mg/dL cutoff compares the models'
*biases*, not their skill: the high-recall models are not more insightful, they
simply alarm more often (linear extrapolation reaches 74% recall by alarming
roughly 21 times a day, which nobody would wear). Second, the fix belongs in the
objective, not the threshold.

So each model emits a **risk score** instead, and the cutoff on that score is
tuned on validation to a false-alarm budget, then applied unchanged to test.
Compared that way, the ranking inverts:

![Recall against false alarms per day](assets/alarm_curve.png)

| at ≤6 false alarms/day | low-glucose recall |
|---|---:|
| **tcn_prob** (predicts a distribution, alarms on P(glucose < 70)) | **74.8%** |
| tcn_cls (head trained directly on the low/not-low label) | 74.2% |
| tcn_hypo3 (loss upweighted toward lows) | 73.2% |
| tcn (plain, best RMSE) | 69.9% |
| persistence | 66.5% |
| linear_extrapolation | 57.8% |

Linear extrapolation goes from apparently best to last. And the model that ships
— `tcn_prob` — catches **five more points of recall than the plain network at
identical accuracy and an identical false-alarm budget**, purely by asking "what
is the probability of going low" instead of "did my one guessed number land
under 70". Same architecture, same 18.9 mg/dL RMSE; only the decision layer
changed.

Full numbers for every model and budget are in [`alarm.md`](alarm.md). Which
model ships was decided on validation patients only — split into two folds, a
threshold tuned on one and scored on the other — because the top three sit
within 1.6 points of each other, and picking among near-ties by reading the test
set turns noise into a decision.

## Does it hold on people who are not like the training set?

Held-out patients answer *does this work on a new person*. They do not answer
*does this work on a different kind of person*: every split above comes from one
corner of the archive — Nightscout exports, overwhelmingly Dexcom.

So the archive's other half became a separate set: **33 patients, 3.9 million
windows** of AndroidAPS exports, a different app with a mix of Dexcom, Medtronic
and Abbott Libre sensors, mostly European. None of it was read when the training
data was assembled. Four donors had uploaded under both formats — one a test
patient, three training patients — and are excluded; without that check this
would have been a re-test on people the model already knew.

The shipped model was applied unchanged, thresholds and all:

| | original test | external population |
|---|---:|---:|
| RMSE | 18.86 | 19.95 |
| RMSE vs persistence | −19% | −16% |
| Clarke A+B | 96.3% | 97.3% |
| recall at ~10 false alarms/day | 74.8% | 69.9% |
| persistence at the same rate | 66.5% | 57.6% |

**The ranking survives and the calibration does not.** At every matched
false-alarm rate the model still beats persistence, by 6 to 12 points. But a
threshold tuned for one false alarm a day delivers 1.8 here, and the six-a-day
setting delivers 10.5 — the same failure that appeared going from validation to
test, in the other direction.

Two independent population shifts broke the threshold and neither broke the
ordering. That points at one design: let the model supply the ordering, and let
a wearer's own first weeks set their cutoff. Per-patient calibration is not a
refinement. Details, including what got worse, are in [`EXTERNAL.md`](EXTERNAL.md).

---

## Why the evaluation is built the way it is

Most of the engineering here went into not fooling ourselves. Three decisions
carry that weight:

**The split is by patient, never by row.** CGM samples five minutes apart are
enormously autocorrelated. Split rows at random and the model can memorise a
trace it has already seen a few steps earlier, which produces a beautiful test
score and a system that fails on the first new person it meets. Splitting whole
patients means the reported number answers the question we actually care about.
The assignment lives in [`artifacts/splits.json`](artifacts) with a fixed seed.

**Baselines come first.** At a 30-minute horizon, glucose is autocorrelated
enough that *persistence* — predicting no change at all — is already a decent
forecaster. Any paper or demo that omits it can make a mediocre model look
impressive. We also include linear extrapolation and ridge regression on the raw
window, so every neural number is read as "how much did the extra complexity
actually buy".

**Lows get their own metrics.** Readings below 70 mg/dL are about 3% of the data.
A model can post an excellent overall RMSE while being useless exactly where a
patient needs it, because the errors on lows are averaged away. So we report RMSE
restricted to lows, the recall and precision of the 30-minute low alarm, the
false-alarm rate per day, and the share of predictions in the clinically
acceptable zones of the Clarke Error Grid.

The demo goes one step further and scores **episodes** rather than readings. One
continuous stretch below 70 is one event a patient experiences; counting each of
its readings separately lets a single long low inflate the score.

---

## Data

The OpenAPS Data Commons is a 6.5 GB archive of Nightscout exports donated by
people running open-source automated insulin delivery. `build_dataset.py` reads
the CGM `entries` records straight out of the zip, keeps sensor glucose only,
clips to a physiologically plausible 20–450 mg/dL, and resamples each patient
onto a regular 5-minute grid.

| | |
|---|---|
| Patients | 40 (of 240 in the archive, ranked by data volume) |
| Readings | 8,145,010 |
| Span | 28,281 patient-days |
| Split | 26 train / 6 validation / 8 test patients |

Two filtering rules matter. Gaps of up to 15 minutes are linearly interpolated;
anything longer invalidates every window that spans it, because filling a
three-hour hole with a straight line invents data and quietly inflates the score.
And a window's prediction target must be a genuinely observed reading — never an
interpolated one.

One patient (`17161370`) holds 1.5 GB across 37 files, an order of magnitude more
than anyone else, and is excluded so a single person cannot dominate training.

---

## Models

All three networks share two design choices:

- **They predict the change, not the level.** The network outputs how far glucose
  will move and we add that to the current reading. Predicting the absolute level
  wastes capacity re-learning "the answer is near where we are now", which
  persistence gives for free.
- **They see the rate of change explicitly.** The first difference of the window
  is a second input channel. A network can derive it, but handing it over
  shortens training and mirrors what a clinician reads off a CGM trace.

| | |
|---|---|
| `persistence` | glucose in 30 min = glucose now |
| `linear_extrapolation` | fit a line to the last 30 min, extend it |
| `ridge` | ridge regression on the raw 24-sample window |
| `lstm` | 2-layer LSTM |
| `tcn` | dilated causal convolutions, 4 levels |
| `transformer` | 3-layer encoder, learned positional embedding |
| `*_hypo{n}` | same architecture, loss upweighted toward lows |
| `ensemble` | mean of the three plain architectures |

The hypo-weighted variants exist to test one question: lows are rare, so does
telling the loss function to care about them more actually buy recall, and what
does it cost in false alarms? `results.md` has the answer.

---

## Repository layout

```
src/
  config.py              all tunable constants
  metrics.py             RMSE/MAE/MARD + hypo alarm + Clarke Error Grid
  predictor.py           checkpoint loading, rolling forecast, episode analysis
  theme.py               chart palette and Plotly chrome
  train.py               training loop, early stopping, checkpointing
  data/
    build_dataset.py     zip -> parquet
    windows.py           supervised windows + patient-level split
  models/
    baselines.py         persistence, linear extrapolation, ridge
    nets.py              LSTM, TCN, Transformer
scripts/
  run_sweep.py           runs everything, writes results.md
app.py                   Streamlit demo
```

---

## Limitations

- **It does not recommend insulin.** The output is a glucose forecast and a
  low-glucose warning. Nothing computes a dose.
- **It has not been tested on a person.** These are retrospective replays of
  recorded traces. No prospective trial, no in-silico closed-loop evaluation.
- **CGM only.** Insulin and carbohydrate records exist in the archive and are not
  used yet, which is why the model is weakest right after meals and corrections —
  the moments its inputs cannot see.
- **One horizon.** 30 minutes. Longer horizons are substantially harder and
  nothing here claims them.
- **No uncertainty.** The model returns a single number with no confidence
  attached, so it cannot yet say when it does not know.

---

## What's next

The limitations list is the roadmap, roughly in order:

1. **Add insulin and carbohydrate inputs.** The archive already contains them.
   This should close most of the post-meal gap.
2. **Predict a distribution, not a point.** A forecast that can express "I don't
   know" is what lets a downstream controller decide when *not* to act.
3. **Withhold predictions on bad input.** Sensor dropouts, artefacts, and traces
   unlike anything in training should produce silence rather than a confident
   wrong number.
4. **Extend to 60 minutes**, and report the horizons separately rather than
   quoting the easy one.

---

## Data use and acknowledgement

CGM traces come from the [OpenAPS Data Commons](https://openaps.org/outcomes/data-commons/),
donated by members of the #WeAreNotWaiting community for research use. The raw
archive is not redistributed in this repository. Thanks to the donors — this does
not exist without them.

## License

MIT
