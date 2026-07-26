"""Round 2: fix the alarm, not the RMSE.

Round 1 produced a clear diagnosis. The networks beat every baseline on RMSE and
then *lost* to persistence on low-glucose recall. That is not a modelling
failure, it is what squared error does: it pulls the forecast toward the mean,
so the model becomes reluctant to commit to the rare extremes, which are
exactly the events the product exists to catch.

Three candidate fixes, trained on the architecture that won round 1:

* ``_prob``, predict a Gaussian instead of a point, and alarm on
  P(glucose < 70). Uncertainty is the thing that lets a forecast say "probably
  fine, but I would not bet on it".
* ``_cls`` , add a head trained directly on the low/not-low label. Optimises
  the decision rather than a number that a decision is later read off.
* ``_mt``  , both at once, plus the loss reweighting from round 1.

All of them are then compared against the baselines at matched false-alarm
budgets, which is the only comparison that is not rigged by threshold choice.

Usage:  python -m scripts.run_round2
"""
from __future__ import annotations

import subprocess
import sys

from src.data.windows import build_windows
from src.metrics import HEADER, format_row
from src.train import TrainConfig, run

EPOCHS = 12
STEPS = 1200


def pick_round1_winner() -> str:
    """Whichever plain architecture won on validation in round 1."""
    import json

    from src.config import ARTIFACTS_DIR

    sweep = ARTIFACTS_DIR / "sweep.json"
    if not sweep.exists():
        return "tcn"
    results = json.loads(sweep.read_text())["results"]
    plain = [r for r in results if r["name"] in ("lstm", "tcn", "transformer")]
    return min(plain, key=lambda r: r["val"]["rmse"])["name"] if plain else "tcn"


def main() -> None:
    arch = pick_round1_winner()
    print(f"Round 1 winner: {arch}. Building windows once...\n", flush=True)
    windows = build_windows(verbose=True)

    variants = [
        ("prob", dict(probabilistic=True)),
        ("cls", dict(classify=True, class_weight=1.0)),
        ("mt", dict(probabilistic=True, classify=True, class_weight=1.0,
                    hypo_weight=3.0)),
    ]

    results = []
    for suffix, kwargs in variants:
        tag = f"{arch}_{suffix}"
        print(f"\n{'=' * 100}\n{tag}\n{'=' * 100}", flush=True)
        results.append(run(
            TrainConfig(model=arch, epochs=EPOCHS, steps_per_epoch=STEPS,
                        tag=tag, **kwargs),
            windows=windows,
        ))

    print(f"\n{'=' * 100}\nROUND 2, forecast accuracy\n{'=' * 100}")
    print(HEADER)
    for r in results:
        print(format_row(f"{r['name']} (test)", r["test"]))

    # Free the big tensors before the alarm report reloads everything.
    del windows, results

    print(f"\n{'=' * 100}\nALARM COMPARISON\n{'=' * 100}", flush=True)
    subprocess.run([sys.executable, "-u", "-m", "scripts.alarm_report"], check=False)


if __name__ == "__main__":
    main()
