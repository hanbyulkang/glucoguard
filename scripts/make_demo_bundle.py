"""Package the smallest thing that still runs the app, for a public deploy.

The full working set is about 110 MB, a 72 MB glucose table plus 39 MB of
cached forecasts, which is more than belongs in a git repository and more than
a free host wants to clone on every deploy.

Almost none of it is needed. The forecasts are already computed, so the hosted
app never has to touch the raw archive: it needs a slice of each wearer's
predictions and a few summary numbers. Taking a contiguous stretch that actually
contains lows keeps the demo honest, it is a real span of a real record, not a
reel of good moments.

Run:  python -m scripts.make_demo_bundle
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, CACHE_DIR, HYPO_THRESHOLD, ROOT, SAMPLE_MINUTES
from src.predictor import load_splits

BUNDLE = ROOT / "demo_data"
N_WEARERS = 4
SLICE_DAYS = 60
MODEL = "tcn_prob"


def pick_slice(frame: pd.DataFrame, days: int) -> pd.DataFrame:
    """The `days`-long stretch of this record containing the most lows."""
    times = pd.to_datetime(frame["target_time"])
    low = (frame["actual"] < HYPO_THRESHOLD).to_numpy()
    span = pd.Timedelta(days=days)

    best_start, best_count = times.iloc[0], -1
    # Step a window forward a week at a time; finer resolution buys nothing here.
    for start in pd.date_range(times.iloc[0], times.iloc[-1] - span, freq="7D"):
        window = (times >= start) & (times < start + span)
        count = int(low[window.to_numpy()].sum())
        if count > best_count:
            best_start, best_count = start, count

    window = (times >= best_start) & (times < best_start + span)
    return frame[window.to_numpy()].reset_index(drop=True)


def main() -> None:
    source = CACHE_DIR / "forecasts" / MODEL
    if not source.exists():
        raise SystemExit(
            f"{source} not found, run `python -m scripts.precompute_forecasts` first."
        )

    BUNDLE.mkdir(exist_ok=True)
    (BUNDLE / "forecasts").mkdir(exist_ok=True)

    test = load_splits()["test"]
    ranked = []
    for pid in test:
        path = source / f"{pid}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        lows = int((frame["actual"] < HYPO_THRESHOLD).sum())
        ranked.append((lows, pid, frame))
    # Wearers with lows to show, but not only the easiest ones, take a spread.
    ranked.sort(reverse=True, key=lambda r: r[0])
    chosen = [ranked[0], ranked[len(ranked) // 3], ranked[2 * len(ranked) // 3],
              ranked[-1]][:N_WEARERS]

    summary = {}
    total = 0
    for lows, pid, frame in chosen:
        cut = pick_slice(frame, SLICE_DAYS)
        keep = [c for c in ("issued_at", "target_time", "predicted", "actual",
                            "current", "sigma", "hypo_prob") if c in cut.columns]
        cut = cut[keep]
        out = BUNDLE / "forecasts" / f"{pid}.parquet"
        cut.to_parquet(out, index=False, compression="zstd")
        size = out.stat().st_size
        total += size

        days = len(cut) * SAMPLE_MINUTES / (60 * 24)
        summary[pid] = {
            "windows": int(len(cut)),
            "days": round(days, 1),
            "time_below_70": float((cut["actual"] < HYPO_THRESHOLD).mean()),
            "lows_in_slice": int((cut["actual"] < HYPO_THRESHOLD).sum()),
            "starts": str(pd.to_datetime(cut["target_time"]).iloc[0].date()),
        }
        print(f"  {pid}: {len(cut):>7,} windows  {days:5.1f} days  "
              f"{summary[pid]['time_below_70']:.2%} low  {size / 1e6:5.2f} MB")

    (BUNDLE / "summary.json").write_text(json.dumps(
        {"model": MODEL, "slice_days": SLICE_DAYS, "wearers": summary}, indent=2))

    checkpoint = ARTIFACTS_DIR / f"{MODEL}.pt"
    if checkpoint.exists():
        shutil.copy2(checkpoint, BUNDLE / f"{MODEL}.pt")
        total += checkpoint.stat().st_size

    for name in ("splits.json", "alarm.json", "selection.json", "matched.json",
                 "policy.json", "calibration.json", "external.json",
                 "multimodal.json", "multimodal_alarm.json", "sweep.json",
                 "thresholds.json", "trajectory.json", "over_time.json",
                 "drift.json", "within_patient.json", "personalized.json"):
        src = ARTIFACTS_DIR / name
        if src.exists():
            shutil.copy2(src, BUNDLE / name)
            total += src.stat().st_size

    print(f"\nWrote {BUNDLE}, {total / 1e6:.1f} MB total, "
          f"{len(chosen)} wearers, {SLICE_DAYS} days each")


if __name__ == "__main__":
    main()
