# Vitalitics 2026 — submission checklist

**Hard deadline: 2026-07-30 (Thu) 21:00 PDT.** The Devpost value is
`2026-07-31T00:00:00-04:00`, which is the *start* of July 31 Eastern — the day
itself never happens. Working to "July 31" is how this gets missed.

Submission page: <https://vitalitics26.devpost.com/challenges/start_a_submission>
Both team members must **Join the hackathon** on Devpost before either can be
added to a submission.

---

## What has to be handed in

Rules say only two things are mandatory: a video and a link to the code.

- [ ] **Demo video, 5 minutes maximum**
  - [ ] YouTube or Google Drive only — those are the two the rules name
  - [ ] YouTube: set to **Public or Unlisted**, never Private
  - [ ] Render under **4:45** so a slow encode cannot push it over
  - [ ] Verify the link opens in a logged-out incognito window
  - [ ] Script: [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md)
- [ ] **Public source link**
  - [ ] `git remote add origin …` and push
  - [ ] Repository set to **Public**
  - [ ] README renders — check the figures resolve on GitHub, not just locally
  - [ ] Confirm `data/` and `artifacts/*.pt` stayed ignored (the archive is not ours to redistribute)
- [ ] **Devpost project page** — copy is in [`DEVPOST.md`](DEVPOST.md)
  - [ ] Elevator pitch (139 characters, fits the ~200 limit)
  - [ ] Inspiration / What it does / How we built it / Challenges / Accomplishments / What we learned / What's next
  - [ ] Built With tags
  - [ ] At least one image — `assets/tradeoff.png` is the strongest single frame
- [ ] Add the second team member as a collaborator
- [ ] **Press Submit.** A saved draft is not a submission. This is the most
      common way a finished project scores zero.

---

## Timeline to the deadline

| When | What |
|---|---|
| now | push the repo public, verify from a logged-out browser |
| now | write the Devpost page and save it as a draft |
| 7/29 | record the screen; leave the day free for a re-take |
| 7/30 morning | edit, render under 4:45, upload, verify the link |
| **7/30 14:00 PDT** | fill the form, add collaborator, **click Submit** |
| **7/30 18:00 PDT** | re-check every link from a logged-out window; fix and resubmit if needed |
| ~~7/30 21:00~~ | entries close |

Seven hours of margin is deliberate. Uploads fail.

---

## Which model to demo

**`tcn_prob`** — the probabilistic TCN, CGM input only.

It was selected on validation wearers alone, using two patient folds, because
the top three candidates sat within 1.6 points of each other and choosing among
near-ties by reading the test set turns noise into a decision.

The variant with insulin and carbohydrate inputs scores 1–2 points better on the
pooled alarm curve. We are not demoing it, for three reasons worth saying aloud
if a judge asks:

- once each wearer has a personal threshold the advantage disappears (77.6%
  against 77.4%);
- it needs treatment records at inference, and recording habits vary 66-fold
  across wearers;
- it makes test RMSE *worse*, so shipping it would mean defending a model that
  is less accurate and barely more useful.

---

## Judging criteria, and where each is answered

Winners are picked on the mean of five criteria. Best Technical Execution is the
single-criterion award and the one to aim at.

| Criterion | Where it is answered |
|---|---|
| Relevancy | CGM → 30-minute forecast → hypo warning is the intersection they describe, exactly |
| **Technical Execution** | patient-level splits, baselines first, an external cohort, four self-caught evaluation errors — all in `Method` |
| Presentation | nine-page app, two figures, a video that opens on a real nocturnal low |
| Innovation | the alarm as a tunable decision, and per-wearer thresholds |
| Impact | 77% of low episodes warned with 25 minutes of median lead |

---

## Rules worth re-reading before submitting

- **Prizes are entirely non-cash** — domains, SaaS licences, certificates. The
  advertised "$6,090" is a list price, not money.
- The organiser's FAQ says all coding must happen during the hackathon; the
  Devpost rules page says nothing of the sort. The two documents disagree. Our
  commits are all inside the window, and `DEVPOST.md` states plainly that the
  background reading predates it.
- Students only, ages 13+. Both of us qualify.
- Publishing the repository is a public disclosure. Nothing about the wearable
  hardware architecture is in it — only the forecasting and alarm layers.

---

## Before you press Submit

- [ ] Video is under 5:00 and plays logged-out
- [ ] Repo is public and README figures render on GitHub
- [ ] No patient data and no model weights in the repo
- [ ] Devpost page has an image
- [ ] Teammate added
- [ ] **Status shows "Submitted", not "Draft"**
