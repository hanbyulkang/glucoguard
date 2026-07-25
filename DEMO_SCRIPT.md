# Demo video script — 5:00 max, target 4:30

Numbers in `{{braces}}` are placeholders filled from `results.md` once the sweep
finishes. Record the screen at 1920×1080; the Streamlit app is laid out for that
width.

---

## 0:00 – 0:40 · The problem

> *(Screen: the app, on a patient's day that contains a nocturnal low. Don't
> touch anything yet — let the chart sit.)*

"This is one night in the life of someone with type 1 diabetes.

Around 2 a.m., their blood sugar starts falling. At 3:15 it crosses 70 —
hypoglycaemia. Their continuous glucose monitor alarms, and that's the problem:
it alarms *when it happens*. By then they're already low, and they're asleep.

A CGM is a very good sensor and a very late alarm. It tells you where you are.
It doesn't tell you where you're going."

---

## 0:40 – 1:20 · What we built

> *(Screen: point at the dashed orange line.)*

"GlucoGuard forecasts blood glucose 30 minutes ahead.

The blue line is what actually happened. The orange dashed line is what our
model predicted — and every point on it was produced half an hour before the
moment it describes, using only data available at that time. So the gap between
the two lines is the error a patient would really have experienced.

The red dots are the moments the model called the low before it started. On this
patient, it caught {{caught_share}} of separate low episodes, with a median
warning of {{median_lead}} minutes."

---

## 1:20 – 2:20 · The data and the honest split

> *(Screen: README data table, then artifacts/splits.json.)*

"We trained on the OpenAPS Data Commons — real CGM traces donated by people
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

## 2:20 – 3:20 · Results, including what doesn't work

> *(Screen: results.md table.)*

"Here's every model we ran, and the baselines come first on purpose.

Persistence — just assuming glucose doesn't change — gets {{persistence_rmse}}
mg/dL. At a 30-minute horizon that's already decent, which is exactly why a
paper that omits it can make a mediocre model look impressive.

Our best model, {{best_model}}, gets {{best_rmse}}. That's {{improvement}}
better.

But look at this row. Linear extrapolation has *worse* overall RMSE —
{{linear_rmse}} — and much *higher* low-glucose recall: {{linear_recall}}
versus persistence's {{persistence_recall}}. It over-predicts the fall, so it
catches more lows and cries wolf more often.

That trade-off is the whole clinical problem, and it's why we don't report a
single accuracy number. We report RMSE restricted to lows, recall, precision,
false alarms per day, and the Clarke Error Grid."

---

## 3:20 – 4:00 · Where the accuracy comes from

> *(Screen: scroll to the banded bar chart.)*

"Lows are about 3% of the data. A model can post an excellent overall RMSE while
being useless exactly where a patient needs it, because the errors on lows get
averaged away.

So we break the error down by what glucose actually did. Below 70, in range,
above 180. This is the chart that tells you whether the model is good or just
lucky.

And we score **episodes**, not readings. One continuous low is one event a
person experiences — counting each of its readings separately lets a single
long low inflate the number."

---

## 4:00 – 4:30 · What it doesn't do, and what's next

> *(Screen: the "What this does not do" expander, opened.)*

"Three things this does not do.

It does not recommend insulin. It's a forecast and a warning, nothing computes
a dose.

It has not been tested on a person — this is a retrospective replay.

And it doesn't know when it doesn't know. It returns one number with no
confidence attached.

That last one is next. The archive already has insulin and carbohydrate records
we're not using yet, and a model that can say 'I'm not sure' is what lets a
safety layer decide when *not* to act.

Everything's reproducible from a public repo: build the dataset, run the sweep,
launch the app. Three commands."

---

## Recording checklist

- [ ] Close every other window; hide the bookmarks bar
- [ ] Browser zoom at 100%, full screen
- [ ] Pick the patient/day in advance — do not hunt on camera
- [ ] Toggle "Overlay persistence baseline" once, on camera, during the results section
- [ ] Render under **4:45** for safety margin against the 5:00 limit
- [ ] Upload to YouTube as **Unlisted or Public** — never Private
- [ ] Verify the link opens in a logged-out incognito window
