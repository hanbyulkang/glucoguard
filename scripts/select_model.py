"""Choose which model ships, using validation patients only.

The first version of this picked the winner by reading `alarm.json`, whose
recall figures are computed on **test**. The top three models sit within 1.6
percentage points of each other, which is exactly the regime where selecting on
the test set turns noise into a decision and quietly inflates the number you
then report as held-out performance.

This selects properly. The six validation patients are split into two folds by
patient; a threshold tuned on one fold is scored on the other, both ways, and the
average is the selection signal. No test data is involved until the winner is
already chosen.

Usage:  python -m scripts.select_model
"""
from __future__ import annotations

import json

import numpy as np
import torch

from src.alarm import alarm_metrics, tune_threshold
from src.config import ARTIFACTS_DIR
from src.data.windows import build_windows
from src.models.baselines import RidgeBaseline, linear_extrapolation, persistence
from src.train import pick_device
from scripts.alarm_report import BUDGETS, load_model_predictions, score_of
from src.alarm import risk_score

SEED = 1337


def val_folds(patient_ids: np.ndarray) -> list[np.ndarray]:
    """Two disjoint masks over validation windows, split by patient."""
    patients = sorted(set(patient_ids))
    rng = np.random.default_rng(SEED)
    shuffled = list(patients)
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    groups = [set(shuffled[:half]), set(shuffled[half:])]
    return [np.isin(patient_ids, list(g)) for g in groups]


def held_out_val_recall(y: np.ndarray, score: np.ndarray,
                        folds: list[np.ndarray]) -> dict[float, float]:
    """Tune on one validation fold, score on the other, average both ways."""
    out = {}
    for budget in BUDGETS:
        recalls = []
        for tune_mask, eval_mask in [(folds[0], folds[1]), (folds[1], folds[0])]:
            thr = tune_threshold(y[tune_mask], score[tune_mask], budget)
            recalls.append(alarm_metrics(y[eval_mask], score[eval_mask], thr)["recall"])
        out[budget] = float(np.mean(recalls))
    return out


def main() -> None:
    device = pick_device()
    windows = build_windows(verbose=False)
    train, val, test = windows["train"], windows["val"], windows["test"]
    folds = val_folds(val.patient_ids)
    print(f"validation folds: {[int(f.sum()) for f in folds]} windows "
          f"({len(set(val.patient_ids))} patients split 3/3)\n")

    Xva = torch.from_numpy(val.X).to(device)
    Xte = torch.from_numpy(test.X).to(device)

    scores: dict[str, np.ndarray] = {}
    scores["persistence"] = risk_score(persistence(val.X))
    scores["linear_extrapolation"] = risk_score(linear_extrapolation(val.X))
    ridge = RidgeBaseline(alpha=1.0).fit(train.X, train.y)
    scores["ridge"] = risk_score(ridge.predict(val.X))

    for path in sorted(ARTIFACTS_DIR.glob("*.pt")):
        name = path.stem
        if name.startswith(("smoke", "_")):
            continue
        pv, _pt, cfg = load_model_predictions(name, Xva, Xte, device)
        scores[name] = score_of(pv, cfg)

    ranked = {n: held_out_val_recall(val.y, s, folds) for n, s in scores.items()}
    order = sorted(ranked, key=lambda n: -np.mean(list(ranked[n].values())))

    header = f"{'model':<24s}" + "".join(f"{f'≤{b:g} FA/day':>14s}" for b in BUDGETS) + f"{'mean':>9s}"
    print("VALIDATION-ONLY selection (tuned on one fold, scored on the other)")
    print(header)
    for name in order:
        row = f"{name:<24s}" + "".join(f"{ranked[name][b]:13.1%} " for b in BUDGETS)
        print(row + f"{np.mean(list(ranked[name].values())):8.1%}")

    winner = order[0]
    payload = {
        "selected": winner,
        "method": "validation split into two patient folds; threshold tuned on one, "
                  "recall measured on the other, averaged both ways. Test data "
                  "was not consulted.",
        "val_recall": ranked,
    }
    with open(ARTIFACTS_DIR / "selection.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    alarm_path = ARTIFACTS_DIR / "alarm.json"
    if alarm_path.exists():
        test_side = json.loads(alarm_path.read_text()).get(winner, {}).get("budgets", {})
        print(f"\nSelected on validation: {winner}")
        print("Its test numbers (reported, not used for the choice):")
        for b, m in test_side.items():
            print(f"  ≤{b:<8s} recall {m['recall']:6.1%}  precision {m['precision']:5.1%}  "
                  f"achieved {m['false_alarms_per_day']:4.1f} FA/day")


if __name__ == "__main__":
    main()
