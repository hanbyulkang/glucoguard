# Demo video script

**Target 4:30, hard limit 5:00.** Record at 1920x1080. Every number below is what
the app actually shows, checked against the deployed build.

Site: <https://glucoguard-oneloop.streamlit.app>

---

## Before you hit record

- [ ] Open the site and click through every page once. The free tier sleeps and
      takes 30 to 60 seconds to wake, and a cold start on camera is 40 wasted
      seconds.
- [ ] Browser at 100% zoom, full screen, bookmarks bar hidden, notifications off.
- [ ] Rehearse the opening state: **The app** > wearer `sim-A` > **Next,
      calibrate** > slider at **6/day** > **Next, monitor**. Confirm the phone
      reads about 127 mg/dL at 06:25. Then return to step 1 and leave it there.
- [ ] Keep a second tab on **Method**, so the closing section is one click away.

**Do not hunt for anything on camera.** Every click below is one you have already
rehearsed.

---

## 0:00 to 0:35 | The problem

**Screen:** step 1 of The app, nothing clicked.

> "Someone with type 1 diabetes wears a continuous glucose monitor, and it tells
> them their blood sugar right now.
>
> The problem is that word, now. When a CGM alarms for hypoglycemia you are
> already hypoglycemic. At three in the morning that alarm is competing with
> sleep, and severe overnight lows are one of the things people with type 1 fear
> most.
>
> The gap is not sensing. It is anticipation. So we asked how much warning you
> can pull out of the signal the device is already producing, with no new
> hardware and no extra inputs."

**Action:** none. Let the page sit.

---

## 0:35 to 1:15 | Setup, and the honesty up front

**Screen:** still step 1.

**Action:** move the cursor across the four tiles, then rest it on the italic
caption underneath.

> "GlucoGuard reads the last two hours and predicts where glucose will be thirty
> minutes from now.
>
> Two things before I show it working. This wearer is simulated. The model is
> real, trained on twenty-eight thousand patient-days of donated CGM traces, and
> everything you see it do is genuine, including its mistakes. The trace is
> generated because the donated recordings are not ours to republish. Every
> measured result I quote later comes from the real cohort.
>
> And it never recommends an insulin dose. It is a forecast and a warning."

**Action:** click **Next, calibrate**.

---

## 1:15 to 2:05 | The dial, which is the actual idea

**Screen:** step 2, slider at 6/day.

**Action:** drag the false-alarm slider slowly from 6 down to 1, hold two
seconds, then back to 6. Let the tiles recompute each time.

> "Here is the part I want to show you.
>
> Catching more lows always costs more false alarms. No setting escapes that
> trade, so instead of hiding it, GlucoGuard hands it over: how many times a day
> are you willing to be interrupted for nothing?
>
> Watch the numbers move. Fewer false alarms, fewer lows caught. More false
> alarms, more lows caught. Same model, same prediction, different answer to a
> question only the wearer can answer.
>
> And the threshold that delivers your setting is fitted to you, from your own
> first two weeks. This wearer lands at twenty percent. Another one in the
> dropdown lands at seventeen. Across our real cohort the spread ran from under
> one percent to nearly eighteen, which is why a single shared cutoff cannot
> work."

**Say this, do not skip it.** The tiles will read about 12.7 false alarms a day
against a target of 6:

> "You will notice it delivers about twice the rate that was asked for. That is
> real and it is on the Calibration page. A threshold fitted on two weeks
> overshoots on the months after it. Per-wearer calibration fixes the spread
> between people, not the absolute level."

**Action:** click **Next, monitor**.

---

## 2:05 to 2:55 | It runs by itself

**Screen:** step 3. Phone shows about 127 mg/dL, green, clock at 06:25.

**Action:** click **Start monitoring**, then stop talking for four or five
seconds and let the clock visibly advance.

> "Now nobody presses anything.
>
> It is reading every five minutes of a recorded day at twelve times speed. The
> counter is climbing, and glucose is drifting down."

**Action:** keep it running. About 17 seconds of playback in, the trace crosses
into a real low, the phone turns red, and a notification card appears.

> "There. The screen went red half an hour of simulated time before the low
> actually arrived, and that card is what would have reached a phone.
>
> Now watch what it does next, which is nothing. It stays quiet for thirty
> minutes even though glucose is still low, because a low that lasts an hour is
> one event, not twelve, and alarming every five minutes is how you manufacture
> alarm fatigue. It speaks again after an hour, because by then silence is
> indistinguishable from the app having crashed."

**Action:** scroll down a little to show the **Alert log** filling in.

---

## 2:55 to 3:45 | The finding that changed the project

**Screen:** click **Models** in the sidebar, scroll to the scatter plot.

**Action:** trace the diagonal with the cursor, left to right.

> "Every model we trained, scored on eight wearers held out completely.
>
> The x-axis is accuracy, lower is better. The y-axis is how many lows each one
> catches, higher is better. Look at the shape. They go the wrong way. Our most
> accurate model is the worst at catching lows. Worse than doing nothing at all.
>
> That is squared error doing exactly what it is supposed to. It rewards a
> forecast that stays near the average, and hypoglycemia is the tail, so
> optimizing accuracy taught the model not to commit to the events we built it
> for. Had we selected on RMSE, the way most work in this area does, we would
> have shipped the worst available alarm."

**Action:** click **Alarm** in the sidebar.

> "And when we compared alarms at the same false-alarm rate instead of a fixed
> seventy cutoff, the ranking inverted. Linear extrapolation looked best at
> seventy-four percent recall. It got there by alarming twenty-one times a day.
> Matched fairly, it comes last."

---

## 3:45 to 4:15 | Does it survive a stranger

**Screen:** click **Generalisation**.

> "Held-out wearers tell you it works on a new person. They do not tell you it
> works on a different kind of person.
>
> So we built a second cohort from the untouched half of the archive: thirty-three
> wearers, a different app, Medtronic and Abbott sensors, mostly European. While
> assembling it we found four donors who had uploaded under both export formats,
> one of them a test wearer. Without that check our external validation would
> have been a re-test on people the model already knew.
>
> Applied unchanged: RMSE nineteen point nine against persistence at twenty-three
> point eight, Clarke A plus B of ninety-seven percent. The ranking survives. The
> calibration does not, which is what sent us to per-wearer thresholds."

---

## 4:15 to 4:40 | What we got wrong

**Screen:** click **Method**, scroll to the mistakes table.

> "Last thing, and it is the one I would most want a judge to see.
>
> This table is every place our own evaluation lied to us and we caught it.
> Reading recall at a fixed cutoff. Choosing the shipped model by peeking at the
> test set. A lead-time metric that counted warnings issued after glucose had
> already dropped.
>
> The last row is the sharpest. Adding insulin and carbohydrate records made
> accuracy worse and the alarm better. We read the accuracy column, concluded the
> extra inputs had failed, and withdrew that an hour later. It is the exact
> mistake this project is about, committed by us, on our own work.
>
> Code and live demo are linked below. Thanks for watching."

---

## Recording checklist

- [ ] Render under **4:45**
- [ ] Upload to YouTube as **Public or Unlisted**, never Private
- [ ] Open the link in a logged-out incognito window and confirm it plays
- [ ] Paste into Devpost, then **press Submit**. A saved draft is not a submission

## If you run long

Cut in this order. These are the least load-bearing:

1. Generalisation, down to one sentence
2. The alert-log scroll at 2:55
3. The slider sweep, down to a single move from 6 to 1

Never cut the Models scatter plot or the mistakes table. Those two are what
separate this from a project that fit a curve.
