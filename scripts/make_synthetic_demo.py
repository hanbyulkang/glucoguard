"""Build the public demo from synthetic wearers instead of donated traces.

The first version of the deploy bundle shipped four real wearers' CGM readings.
That was a mistake: the OpenAPS Data Commons is donated patient data behind a
use agreement, redistribution is not obviously permitted, and a public
repository cannot be un-published — forks and caches outlive any deletion.

Aggregate findings are a different matter. RMSE tables, alarm curves and
calibration statistics are *results*, not data, and those still come from the
real cohort. What the hosted app plays back is simulated.

The simulation is deliberately unflattering. It is not a smooth curve the model
will ace; it carries meal spikes, correction overshoot, dawn rise, sensor noise
and dropouts, and it is tuned to put a wearer below 70 several times a week.
The model is the real trained model, running on it unmodified — so the demo
shows genuine behaviour, including genuine mistakes, on a patient who does not
exist.

Run:  python -m scripts.make_synthetic_demo
"""
from __future__ import annotations

import json
import re
import shutil

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, HYPO_THRESHOLD, ROOT, SAMPLE_MINUTES
from src.predictor import Forecaster, rolling_forecast

BUNDLE = ROOT / "demo_data"
DAYS = 45
MODEL = "tcn_prob"
START = pd.Timestamp("2024-03-01T00:00:00Z")

# Four simulated wearers with deliberately different habits, so the per-wearer
# calibration step has something real to distinguish. Each carries a target for
# how much time it should spend under 70, spanning the range the real cohort
# covers (roughly 1.5% to 7%).
#
# `baseline` is solved for at build time rather than written down, because a
# hand-tuned constant does not survive a change to any other parameter: an
# earlier version had numbers fitted against a 20-day run that produced 25% time
# below 70 over 45 days, which would have shipped a demo of a wearer in
# permanent hypoglycaemia.
PROFILES = {
    "sim-A": dict(target_tbr=0.055, meal_size=(45, 90), meals=3, overshoot=0.56,
                  dawn=18, noise=5.5, seed=11, label="tight control, frequent lows"),
    "sim-B": dict(target_tbr=0.018, meal_size=(35, 70), meals=4, overshoot=0.40,
                  dawn=25, noise=6.5, seed=22, label="runs high, rare lows"),
    "sim-C": dict(target_tbr=0.042, meal_size=(55, 110), meals=3, overshoot=0.65,
                  dawn=12, noise=7.5, seed=33, label="large swings"),
    "sim-D": dict(target_tbr=0.028, meal_size=(30, 60), meals=5, overshoot=0.50,
                  dawn=8, noise=4.5, seed=44, label="steady, grazes"),
}


def simulate(profile: dict, days: int) -> pd.DataFrame:
    """A glucose trace with the shapes a CGM actually produces.

    Not a physiological model — a curve generator whose failure modes resemble
    the real ones: a meal ramps over ~40 minutes and decays over hours, the
    correction that follows overshoots downward, glucose drifts up before dawn,
    and the sensor adds noise on top of all of it.
    """
    rng = np.random.default_rng(profile["seed"])
    n = int(days * 24 * 60 / SAMPLE_MINUTES)
    t = np.arange(n)
    minutes = t * SAMPLE_MINUTES
    hour = (minutes / 60) % 24

    glucose = np.full(n, float(profile["baseline"]))

    # Dawn phenomenon: a rise through the small hours, peaking around 07:00.
    glucose += profile["dawn"] * np.exp(-((hour - 6.5) ** 2) / 6.0)

    # Meals, and the correction that chases each one.
    lo, hi = profile["meal_size"]
    for day in range(days):
        for meal in range(profile["meals"]):
            centre = day * 24 + rng.normal(7 + meal * 5, 0.9)
            if not 0 <= centre < days * 24:
                continue
            rise = rng.uniform(lo, hi)
            peak_at = centre * 60 / SAMPLE_MINUTES
            width = rng.uniform(6, 11)

            ramp = rise * np.exp(-((t - peak_at) ** 2) / (2 * width**2))
            # The insulin response lands late and pulls further than it should.
            lag = width * rng.uniform(2.4, 3.6)
            fall = (rise * profile["overshoot"]
                    * np.exp(-((t - peak_at - lag) ** 2) / (2 * (width * 1.7) ** 2)))
            glucose += ramp - fall

    # Slow wander, then sensor noise on top.
    wander = np.cumsum(rng.normal(0, 0.55, n))
    wander -= np.linspace(0, wander[-1], n)
    glucose += wander
    glucose += rng.normal(0, profile["noise"], n)
    glucose = np.clip(glucose, 38, 380)

    times = START + pd.to_timedelta(minutes, unit="m")
    frame = pd.DataFrame({"datetime": times, "glucose": glucose.astype(np.float32)})

    # Real CGM drops out. Punch a few holes so the app's refusal path is live.
    for _ in range(int(days / 6)):
        start = rng.integers(0, n - 40)
        frame.loc[start : start + rng.integers(4, 36), "glucose"] = np.nan
    return frame


ID_PATTERN = re.compile(r"^(?:aaps_)?\d{6,}$")


def anonymise(node, mapping: dict):
    """Replace archive patient ids with stable pseudonyms, everywhere they occur.

    The results carry per-person figures — a wearer's own alarm threshold and
    how much time they spend below 70. Those are findings worth publishing, but
    keyed by the archive's own identifier they would link a published health
    statistic back to a specific donated record. The pseudonym keeps the finding
    and drops the link.
    """
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            new_key = mapping.get(key, key) if isinstance(key, str) else key
            out[new_key] = anonymise(value, mapping)
        return out
    if isinstance(node, list):
        return [anonymise(v, mapping) for v in node]
    if isinstance(node, str):
        return mapping.get(node, node)
    return node


def build_id_map() -> dict:
    """W1.. for the held-out cohort, E1.. for the external one, T1.. for training."""
    splits = json.loads((ARTIFACTS_DIR / "splits.json").read_text())
    mapping = {}
    for prefix, key in (("T", "train"), ("V", "val"), ("W", "test")):
        for i, pid in enumerate(sorted(splits.get(key, [])), 1):
            mapping[pid] = f"{prefix}{i}"

    seen = set()
    for name in ("thresholds.json", "trajectory.json", "personalized.json",
                 "within_patient.json", "calibration.json"):
        path = ARTIFACTS_DIR / name
        if not path.exists():
            continue
        stack = [json.loads(path.read_text())]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for k, v in node.items():
                    if isinstance(k, str) and ID_PATTERN.match(k):
                        seen.add(k)
                    stack.append(v)
            elif isinstance(node, list):
                stack.extend(node)
            elif isinstance(node, str) and ID_PATTERN.match(node):
                seen.add(node)
    for i, pid in enumerate(sorted(seen - set(mapping)), 1):
        mapping[pid] = f"E{i}"
    return mapping


def solve_baseline(profile: dict, days: int, tolerance: float = 0.003) -> float:
    """Bisect on the baseline until the trace spends the intended time low.

    Time below 70 falls monotonically as the baseline rises, so bisection is
    enough and converges in a couple of dozen evaluations.
    """
    target = profile["target_tbr"]
    lo, hi = 70.0, 210.0
    best = (hi + lo) / 2

    for _ in range(28):
        best = (lo + hi) / 2
        trace = simulate({**profile, "baseline": best}, days)
        glucose = trace["glucose"].dropna()
        tbr = float((glucose < HYPO_THRESHOLD).mean())
        if abs(tbr - target) <= tolerance:
            return best
        if tbr > target:
            lo = best          # too many lows: raise the baseline
        else:
            hi = best
    return best


def main() -> None:
    checkpoint = ARTIFACTS_DIR / f"{MODEL}.pt"
    if not checkpoint.exists():
        raise SystemExit(f"{checkpoint} not found — train a model first.")

    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    (BUNDLE / "forecasts").mkdir(parents=True)

    fc = Forecaster(MODEL)
    summary, total = {}, 0

    for name, profile in PROFILES.items():
        baseline = solve_baseline(profile, DAYS)
        profile = {**profile, "baseline": baseline}
        series = simulate(profile, DAYS)
        frame = rolling_forecast(series, fc)
        keep = [c for c in ("issued_at", "target_time", "predicted", "actual",
                            "current", "sigma", "hypo_prob") if c in frame.columns]
        frame = frame[keep]

        out = BUNDLE / "forecasts" / f"{name}.parquet"
        frame.to_parquet(out, index=False, compression="zstd")
        total += out.stat().st_size

        tbr = float((frame["actual"] < HYPO_THRESHOLD).mean())
        summary[name] = {
            "windows": int(len(frame)),
            "days": round(len(frame) * SAMPLE_MINUTES / (60 * 24), 1),
            "time_below_70": tbr,
            "lows_in_slice": int((frame["actual"] < HYPO_THRESHOLD).sum()),
            "label": profile["label"],
            "synthetic": True,
        }
        print(f"  {name}: {len(frame):>7,} windows  {summary[name]['days']:5.1f} days  "
              f"{tbr:6.2%} low (target {profile['target_tbr']:.1%}, "
              f"baseline solved to {baseline:.0f})  {profile['label']}")

    (BUNDLE / "summary.json").write_text(json.dumps(
        {"model": MODEL, "synthetic": True, "days": DAYS,
         "note": "Simulated wearers. The model is the real trained checkpoint; "
                 "the patients are not real and no donated data is redistributed.",
         "wearers": summary}, indent=2))

    (BUNDLE / "splits.json").write_text(json.dumps(
        {"seed": 0, "train": [], "val": [], "test": sorted(PROFILES)}, indent=2))

    shutil.copy2(checkpoint, BUNDLE / f"{MODEL}.pt")
    total += checkpoint.stat().st_size

    # Aggregate findings are results, not data, and stay real — but the ones
    # keyed by wearer get pseudonymised on the way out.
    id_map = build_id_map()
    for name in ("alarm.json", "selection.json", "matched.json", "policy.json",
                 "calibration.json", "external.json", "multimodal.json",
                 "multimodal_alarm.json", "sweep.json", "thresholds.json",
                 "trajectory.json", "over_time.json", "drift.json",
                 "within_patient.json", "personalized.json"):
        src = ARTIFACTS_DIR / name
        if not src.exists():
            continue
        payload = anonymise(json.loads(src.read_text()), id_map)
        dest = BUNDLE / name
        dest.write_text(json.dumps(payload, indent=2))
        total += dest.stat().st_size
    print(f"  pseudonymised {len(id_map)} wearer ids in the published results")

    print(f"\nWrote {BUNDLE} — {total / 1e6:.1f} MB, {len(PROFILES)} simulated wearers")
    print("No donated patient readings are included.")


if __name__ == "__main__":
    main()
