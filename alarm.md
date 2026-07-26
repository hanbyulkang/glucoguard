# Low-glucose alarm: matched false-alarm comparison

Reporting recall at a fixed 70 mg/dL cutoff compares the *biases* of differently-trained predictors, not their skill. A model trained on squared error is pulled toward the mean and under-shoots rare lows; linear extrapolation over-shoots every fall. Read at one cutoff, the over-shooting model looks like the better alarm purely because it alarms more often.

So each model instead emits a risk score, and the threshold on that score is tuned on the **validation** split to hit a false-alarm budget. The same threshold is then applied to **test**, unchanged.

| model | recall @ ≤1 FA/day | precision | achieved FA/day | recall @ ≤3 FA/day | precision | achieved FA/day | recall @ ≤6 FA/day | precision | achieved FA/day |
|---|---|---|---|---|---|---|---|---|---|
| tcn_prob | 38.3% | 59.2% | 3.3 | 59.9% | 47.8% | 8.1 | 74.8% | 38.6% | 14.7 |
| tcn_cls | 36.6% | 58.9% | 3.2 | 58.8% | 47.2% | 8.1 | 74.2% | 38.1% | 14.9 |
| tcn_hypo3 | 37.4% | 59.6% | 3.1 | 59.2% | 47.8% | 8.0 | 73.2% | 38.7% | 14.3 |
| tcn_hypo8 | 37.3% | 59.3% | 3.2 | 58.6% | 47.9% | 7.9 | 73.0% | 38.7% | 14.3 |
| tcn_mt | 34.6% | 56.9% | 3.2 | 55.8% | 46.2% | 8.0 | 71.8% | 38.2% | 14.3 |
| ridge | 29.6% | 52.2% | 3.4 | 55.3% | 44.9% | 8.4 | 71.6% | 37.3% | 14.8 |
| transformer | 36.7% | 59.9% | 3.0 | 57.6% | 48.3% | 7.6 | 71.1% | 38.9% | 13.8 |
| tcn | 37.7% | 59.8% | 3.1 | 56.9% | 48.1% | 7.6 | 69.9% | 38.8% | 13.6 |
| lstm | 34.2% | 53.3% | 3.7 | 55.1% | 42.8% | 9.1 | 68.2% | 34.7% | 15.9 |
| persistence | 33.1% | 51.3% | 3.9 | 55.1% | 40.2% | 10.1 | 66.5% | 33.4% | 16.4 |
| linear_extrapolation | 12.4% | 36.9% | 2.6 | 36.7% | 39.4% | 7.0 | 57.8% | 36.1% | 12.6 |

## The achieved false-alarm rate overshoots the budget, and that is the finding

A threshold tuned to 1 false alarm per day on validation delivers several on test. The two splits are not equally hard: the validation patients spend far less time below 70 than the test patients do, so a cutoff calibrated on the first is too permissive for the second.

This is worth stating plainly rather than tuning away. It means a population-level alarm threshold does not transfer between people, and that per-patient calibration, using a wearer's own first weeks of data to set their threshold, is not a refinement but a requirement.
