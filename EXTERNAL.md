# External validation: a different population, no retraining

Held-out patients answer *does this work on a new person*. They do not answer *does this work on a different kind of person*. Every split so far came from one corner of the OpenAPS archive: Nightscout exports, overwhelmingly Dexcom sensors.

So the archive's other half was built into a separate set. Those are AndroidAPS exports, a different app, a mix of Dexcom, Medtronic and Abbott Libre sensors, and UTC offsets that place most of these users in Europe. None of it was read when the training data was assembled.

**33 patients, 3,927,464 windows, 2.14% of them below 70 mg/dL** (the original test patients were at 4.29%, so this population goes low about half as often).

Four donors had uploaded under both export formats, one of them a test patient, three of them training patients. They are excluded. Without that check this would have been a re-test on people the model already knew.

The shipped model was applied unchanged: no retraining, no refitting, and the alarm thresholds are the ones already tuned on the original validation patients.

## The forecast transfers

| | RMSE | MAE | MARD | Clarke A+B |
|---|---:|---:|---:|---:|
| persistence | 23.77 | 16.58 | 12.63% | 96.17% |
| tcn_prob | 19.95 | 13.80 | 10.76% | 97.25% |

RMSE degrades from 18.86 on the original test patients to 19.95 here, about 6% worse, while still beating persistence by 16%. Clinical acceptability is unchanged at 97.3%.

## The alarm transfers, but the threshold does not

| threshold tuned for | recall here | precision | achieved FA/day | persistence at the same FA/day |
|---|---:|---:|---:|---:|
| ≤1/day | 28.8% | 49.2% | 1.8 | 22.6% |
| ≤3/day | 52.6% | 38.0% | 5.3 | 42.8% |
| ≤6/day | 69.9% | 29.2% | 10.5 | 57.6% |

The ranking survives: at every matched false-alarm rate the model beats persistence, by 6 to 12 points. That is the claim this project rests on, and it holds on a population it has never seen.

What does not survive is the *calibration*. A threshold tuned for one false alarm a day delivers 1.8 here; the six-a-day setting delivers 10.5. The same failure appeared going from validation to test, in the other direction. Groups of people differ in how often they go low, and a cutoff fitted to one group is simply the wrong cutoff for another.

## What got worse, plainly

- **RMSE on lows rose from 25.20 to 28.66 mg/dL.** The forecast is less accurate in exactly the region that matters, on people it has not seen.
- **Read at a fixed 70 mg/dL cutoff, recall collapses to 10.2%.** The tuned alarm is doing the work; the raw point forecast alone would be close to useless here.
- Every conclusion above is retrospective replay. Nothing has been tested prospectively, and no one has worn this.

## What this changes

Per-patient calibration stops being a nice-to-have. Two independent population shifts both broke the threshold and neither broke the ranking, which points at the same design: use a wearer's own first weeks to set their cutoff, and let the model supply only the ordering.
