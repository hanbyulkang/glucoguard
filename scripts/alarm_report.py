"""Compare every model as a low-glucose *alarm*, at matched false-alarm budgets.

`run_sweep.py` answers "how accurate is the forecast". This answers the question
a patient would actually ask: *given that I will tolerate this many false alarms
a day, how many of my lows does each system catch?*

Thresholds are tuned on the validation split and applied unchanged to test.

Usage:  python -m scripts.alarm_report
"""
from __future__ import annotations

import json

import numpy as np
import torch

from src.alarm import PREDICTIONS_PER_DAY, pr_curve, recall_at_budget, risk_score
from src.config import ARTIFACTS_DIR, HYPO_THRESHOLD
from src.data.windows import build_windows
from src.models.baselines import RidgeBaseline, linear_extrapolation, persistence
from src.train import as_point, pick_device, predict
from src.models.nets import build as build_net

BUDGETS = (1.0, 3.0, 6.0)


def load_model_predictions(name: str, Xva, Xte, device):
    blob = torch.load(ARTIFACTS_DIR / f"{name}.pt", map_location="cpu",
                      weights_only=False)
    cfg, norm = blob["config"], blob["norm"]
    model = build_net(cfg["model"], mean=norm["mean"], std=norm["std"],
                      heteroscedastic=cfg.get("probabilistic", False),
                      classify=cfg.get("classify", False)).to(device)
    model.load_state_dict(blob["state_dict"])
    return predict(model, Xva), predict(model, Xte), cfg


def score_of(pred: np.ndarray, cfg: dict) -> np.ndarray:
    """Pick the most informative risk signal each model is able to give.

    A trained classifier head beats both alternatives when present: it was
    optimised for this exact decision. Thresholding its raw logit is equivalent
    to thresholding its probability, so no sigmoid is needed here.
    """
    if cfg.get("classify"):
        return pred[..., -1]
    if cfg.get("probabilistic"):
        return risk_score(pred[..., 0], pred[..., 1])
    return risk_score(as_point(pred))


def main() -> None:
    device = pick_device()
    windows = build_windows(verbose=False)
    train, val, test = windows["train"], windows["val"], windows["test"]
    Xva = torch.from_numpy(val.X).to(device)
    Xte = torch.from_numpy(test.X).to(device)

    scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # --- baselines -----------------------------------------------------------
    scores["persistence"] = (risk_score(persistence(val.X)),
                             risk_score(persistence(test.X)))
    scores["linear_extrapolation"] = (risk_score(linear_extrapolation(val.X)),
                                      risk_score(linear_extrapolation(test.X)))
    ridge = RidgeBaseline(alpha=1.0).fit(train.X, train.y)
    scores["ridge"] = (risk_score(ridge.predict(val.X)),
                       risk_score(ridge.predict(test.X)))

    # --- every trained checkpoint -------------------------------------------
    for path in sorted(ARTIFACTS_DIR.glob("*.pt")):
        name = path.stem
        if name.startswith(("smoke", "_")):
            continue
        pv, pt, cfg = load_model_predictions(name, Xva, Xte, device)
        scores[name] = (score_of(pv, cfg), score_of(pt, cfg))
        print(f"scored {name}", flush=True)

    # --- report --------------------------------------------------------------
    report = {}
    for name, (sv, st) in scores.items():
        report[name] = {
            "budgets": recall_at_budget(val.y, sv, test.y, st, BUDGETS),
            "pr_curve_test": pr_curve(test.y, st),
        }

    with open(ARTIFACTS_DIR / "alarm.json", "w") as fh:
        json.dump(report, fh, indent=2)

    header = f"{'model':<24s}" + "".join(
        f"{'  ≤%g FA/day: recall' % b:>26s}" for b in BUDGETS
    )
    print(f"\n{header}")
    ranked = sorted(report, key=lambda n: -report[n]["budgets"][f"{BUDGETS[-1]:g}/day"]["recall"])
    for name in ranked:
        row = f"{name:<24s}"
        for b in BUDGETS:
            m = report[name]["budgets"][f"{b:g}/day"]
            row += f"   {m['recall']:6.1%} (prec {m['precision']:4.0%}, got {m['false_alarms_per_day']:4.1f})"
        print(row)

    write_markdown(report)
    print(f"\nWrote {ARTIFACTS_DIR / 'alarm.json'} and alarm.md")


def write_markdown(report: dict) -> None:
    lines = [
        "# Low-glucose alarm, matched false-alarm comparison",
        "",
        "Reporting recall at a fixed 70 mg/dL cutoff compares the *biases* of "
        "differently-trained predictors, not their skill. A model trained on "
        "squared error is pulled toward the mean and under-shoots rare lows; "
        "linear extrapolation over-shoots every fall. Read at one cutoff, the "
        "over-shooting model looks like the better alarm purely because it "
        "alarms more often.",
        "",
        "So each model instead emits a risk score, and the threshold on that "
        "score is tuned on the **validation** split to hit a false-alarm budget. "
        "The same threshold is then applied to **test**, unchanged.",
        "",
        "| model | " + " | ".join(
            f"recall @ ≤{b:g} FA/day | precision | achieved FA/day" for b in BUDGETS
        ) + " |",
        "|---" * (1 + 3 * len(BUDGETS)) + "|",
    ]
    ranked = sorted(report, key=lambda n: -report[n]["budgets"][f"{BUDGETS[-1]:g}/day"]["recall"])
    for name in ranked:
        cells = [name]
        for b in BUDGETS:
            m = report[name]["budgets"][f"{b:g}/day"]
            cells += [f"{m['recall']:.1%}", f"{m['precision']:.1%}",
                      f"{m['false_alarms_per_day']:.1f}"]
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## The achieved false-alarm rate overshoots the budget, and that is the finding",
        "",
        "A threshold tuned to 1 false alarm per day on validation delivers several "
        "on test. The two splits are not equally hard: the validation patients "
        "spend far less time below 70 than the test patients do, so a cutoff "
        "calibrated on the first is too permissive for the second.",
        "",
        "This is worth stating plainly rather than tuning away. It means a "
        "population-level alarm threshold does not transfer between people, and "
        "that per-patient calibration, using a wearer's own first weeks of data "
        "to set their threshold, is not a refinement but a requirement.",
    ]
    (ARTIFACTS_DIR.parent / "alarm.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
