# Demo video script: 5:00 max, target 4:30

Numbers below are the measured ones from `results.md` and `alarm.md`. Record the
screen at 1920×1080; the Streamlit app is laid out for that width.

---

## 0:00 – 0:40 · The problem

> *(Screen: the app, on a patient's day that contains a nocturnal low. Don't
> touch anything yet, let the chart sit.)*

"This is one night in the life of someone with type 1 diabetes.

Around 2 a.m. their blood sugar starts falling. At 3:15 it crosses 70, 
hypoglycaemia. Their continuous glucose monitor alarms, and that's the problem:
it alarms *when it happens*. By then they're already low, and they're asleep.

A CGM is a very good sensor and a very late alarm. It tells you where you are.
It doesn't tell you where you're going."

---

## 0:40 – 1:20 · What we built

> *(Screen: point at the dashed orange line.)*

"GlucoGuard forecasts blood glucose 30 minutes ahead.

The blue line is what actually happened. The orange dashed line is what our
model predicted, and every point on it was produced half an hour before the
moment it describes, using only data available at that time. So the gap between
the two lines is the error a patient would really have experienced.

The red dots are the moments the model called the low before it started. On this
patient, it caught 27% of separate low episodes, with a median
warning of 30 minutes."

---

## 1:20 – 2:20 · The data and the honest split

> *(Screen: README data table, then artifacts/splits.json.)*

"We trained on the OpenAPS Data Commons, real CGM traces donated by people
running open-source insulin delivery. 40 patients, 8.1 million readings,
28,281 patient-days.

One decision matters more than any modelling choice here. CGM samples five
minutes apart are enormously correlated, so if you split the data randomly, the
model can memorise a trace it saw a few steps earlier. You get a beautiful test
score and a system that fails on the first new person it meets.

So we split by **patient**. 26 people to train, 6 to choose the model, 8 the
model never sees until the very end. Everything you're looking at is from those
8 held-out people."

---

## 2:20 – 3:30 · The finding: accuracy and safety pulled apart

> *(Screen: results.md table.)*

"Here's every model we ran, and the baselines come first on purpose.

Persistence, just assuming glucose doesn't change, gets 23.2
mg/dL. At a 30-minute horizon that's already decent, which is exactly why a
paper that omits it can make a mediocre model look impressive.

Our best model, tcn_prob, gets 18.9 mg/dL. That's 19%
better.

But now rank these models by low-glucose recall instead, and the order almost
exactly reverses. Our most accurate model is the *worst* at catching lows, worse
than doing nothing.

That's not a bug in one model. Squared error rewards a forecast that hugs the
mean, and hypoglycaemia is the tail. Optimising accuracy taught the model to
refuse to commit to the exact events we built it for. Selecting on RMSE would
have shipped the worst available alarm.

And the models that looked good on recall weren't insightful, they just alarmed
more. Linear extrapolation hits 74% recall by firing 21 times a day. Nobody
wears that."

---

## 3:30 – 4:10 · The fix, and the slider that shows it

> *(Screen: back to the app. Move the false-alarm slider from 1/day to 6/day and
> let the risk line and the tiles move.)*

"So we stopped reading the alarm off the forecast. The model now predicts a
*distribution*, and the alarm asks the right question, what's the probability
of going below 70, with a threshold tuned to a false-alarm budget you choose.

This slider is that budget. Drag it right, catch more lows, tolerate more false
alarms. Drag it left, the opposite. That dial doesn't exist with a fixed
70 mg/dL rule.

Compared this way, every model at the *same* false-alarm budget, linear
extrapolation drops from apparently best to last. And the model we ship catches
75% of lows where the plain network gets 70%, at identical accuracy and identical
false alarms. Same architecture, same 18.9 RMSE. Only the decision layer
changed."

---

## 4:10 – 4:40 · What it doesn't do, and what's next

> *(Screen: the "What this does not do" expander, opened.)*

"Three things this does not do.

It does not recommend insulin. It's a forecast and a warning, nothing computes
a dose.

It has not been tested on a person, this is a retrospective replay.

And it's still blind to meals and insulin, which is exactly why it's weakest
after eating. Those records are already in the archive; that's next.

One more honest number. A threshold tuned to one false alarm a day on our
validation patients delivers several on our test patients, because they're
different people who go low at different rates. That means a population-level
alarm threshold doesn't transfer, and per-patient calibration isn't a
refinement, it's a requirement.

Everything's reproducible from a public repo: build the dataset, run the sweep,
launch the app. Three commands."

---

## Recording checklist

- [ ] Close every other window; hide the bookmarks bar
- [ ] Browser zoom at 100%, full screen
- [ ] Pick the patient/day in advance, do not hunt on camera
- [ ] Move the false-alarm slider on camera during section 3:30, it is the demo's best moment
- [ ] Toggle "Overlay persistence baseline" once, on camera
- [ ] Render under **4:45** for safety margin against the 5:00 limit
- [ ] Upload to YouTube as **Unlisted or Public**, never Private
- [ ] Verify the link opens in a logged-out incognito window
