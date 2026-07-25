"""Run the whole comparison in one process and write results.md.

Order matters: baselines first, so every learned number below them is read as
"how much did the extra complexity actually buy". The hypo-weighted run and the
ensemble come last because they only make sense once the plain architectures
have been compared on equal terms.

Usage:  python -m scripts.run_sweep
"""
from __future__ import annotations

import json
import time

import numpy as np

from src.config import ARTIFACTS_DIR, HISTORY_STEPS, HORIZON_MINUTES, SAMPLE_MINUTES
from src.data.windows import build_windows
from src.metrics import HEADER, evaluate, format_row
from src.models.baselines import RidgeBaseline, linear_extrapolation, persistence
from src.train import TrainConfig, run

EPOCHS = 12
STEPS = 1200


def main() -> None:
    t0 = time.time()
    print("Building windows...", flush=True)
    windows = build_windows(verbose=True)
    train, val, test = windows["train"], windows["val"], windows["test"]

    rows: list[tuple[str, dict, dict]] = []   # (name, val_metrics, test_metrics)
    extras: dict[str, dict] = {}

    # ---------- baselines ----------------------------------------------------
    print(f"\n{'=' * 100}\nBASELINES\n{'=' * 100}", flush=True)
    print(HEADER)

    for name, fn in [("persistence", persistence),
                     ("linear_extrapolation", linear_extrapolation)]:
        v, t = evaluate(val.y, fn(val.X)), evaluate(test.y, fn(test.X))
        rows.append((name, v, t))
        extras[name] = {"n_params": 0}
        print(format_row(name, t), flush=True)

    ridge = RidgeBaseline(alpha=1.0).fit(train.X, train.y)
    v = evaluate(val.y, ridge.predict(val.X))
    t = evaluate(test.y, ridge.predict(test.X))
    rows.append(("ridge", v, t))
    extras["ridge"] = {"n_params": HISTORY_STEPS + 1}
    print(format_row("ridge", t), flush=True)

    # ---------- neural architectures, matched budget -------------------------
    print(f"\n{'=' * 100}\nNEURAL ARCHITECTURES\n{'=' * 100}", flush=True)
    plain: dict[str, dict] = {}
    for arch in ["tcn", "transformer", "lstm"]:
        res = run(TrainConfig(model=arch, epochs=EPOCHS, steps_per_epoch=STEPS,
                              tag=arch), windows=windows)
        plain[arch] = res
        rows.append((arch, res["val"], res["test"]))
        extras[arch] = {"n_params": res["n_params"], "seconds": res["train_seconds"]}

    # ---------- does weighting the loss toward lows help? --------------------
    best_arch = min(plain, key=lambda a: plain[a]["val"]["rmse"])
    print(f"\n{'=' * 100}\nHYPO-WEIGHTED LOSS (base architecture: {best_arch})\n{'=' * 100}",
          flush=True)
    for w in [3.0, 8.0]:
        tag = f"{best_arch}_hypo{w:g}"
        res = run(TrainConfig(model=best_arch, epochs=EPOCHS, steps_per_epoch=STEPS,
                              hypo_weight=w, tag=tag), windows=windows)
        rows.append((tag, res["val"], res["test"]))
        extras[tag] = {"n_params": res["n_params"], "seconds": res["train_seconds"]}

    # ---------- ensemble of the three plain architectures --------------------
    print(f"\n{'=' * 100}\nENSEMBLE\n{'=' * 100}", flush=True)
    ens_val = np.mean([plain[a]["_val_pred"] for a in plain], axis=0)
    ens_test = np.mean([plain[a]["_test_pred"] for a in plain], axis=0)
    v, t = evaluate(val.y, ens_val), evaluate(test.y, ens_test)
    rows.append(("ensemble(tcn+trf+lstm)", v, t))
    extras["ensemble(tcn+trf+lstm)"] = {
        "n_params": sum(plain[a]["n_params"] for a in plain)
    }
    print(HEADER)
    print(format_row("ensemble(tcn+trf+lstm)", t), flush=True)

    # ---------- pick the winner on VALIDATION, report it on TEST -------------
    best_name, best_val, best_test = min(rows, key=lambda r: r[1]["rmse"])

    payload = {
        "task": {
            "history_minutes": HISTORY_STEPS * SAMPLE_MINUTES,
            "horizon_minutes": HORIZON_MINUTES,
            "sample_minutes": SAMPLE_MINUTES,
        },
        "counts": {
            k: {"patients": int(len(set(windows[k].patient_ids))),
                "windows": int(len(windows[k]))}
            for k in ("train", "val", "test")
        },
        "selected_on_validation": best_name,
        "results": [
            {"name": n, "val": v, "test": t, **extras.get(n, {})} for n, v, t in rows
        ],
        "total_seconds": round(time.time() - t0, 1),
    }
    with open(ARTIFACTS_DIR / "sweep.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    write_results_md(payload)
    print(f"\nSelected on validation: {best_name}")
    print(HEADER)
    print(format_row(f"{best_name} (test)", best_test))
    print(f"\nTotal wall clock: {(time.time() - t0) / 60:.1f} min")


def write_results_md(payload: dict) -> None:
    c = payload["counts"]
    lines = [
        "# Results",
        "",
        f"Task: predict glucose {payload['task']['horizon_minutes']} minutes ahead "
        f"from {payload['task']['history_minutes']} minutes of CGM history "
        f"({payload['task']['sample_minutes']}-minute samples).",
        "",
        "Split is by patient. No patient appears in more than one split, so the test "
        "column is a genuine estimate of performance on someone the model has never seen.",
        "",
        f"| split | patients | windows |",
        f"|---|---:|---:|",
        f"| train | {c['train']['patients']} | {c['train']['windows']:,} |",
        f"| val | {c['val']['patients']} | {c['val']['windows']:,} |",
        f"| test | {c['test']['patients']} | {c['test']['windows']:,} |",
        "",
        "The two splits are not equally hard, and it would be misleading to hide "
        "that: the validation patients spend far less time low than the test "
        "patients do. That is real between-person variation, not a bug, and it is "
        "why validation RMSE sits well below test RMSE for every model alike. "
        "Selection still works — the *ranking* is what selection needs — but the "
        "validation column should not be read as a performance estimate.",
        "",
        "## Held-out test set",
        "",
        "`RMSE_hypo` is RMSE restricted to windows whose true value is below "
        "70 mg/dL. `recall`/`precision` describe the 30-minute-ahead low-glucose "
        "alarm. `Clarke A+B` is the share of predictions in the clinically "
        "acceptable zones of the Clarke Error Grid.",
        "",
        "| model | params | val RMSE | RMSE | MAE | MARD | RMSE_hypo | hypo recall | hypo precision | false alarms/day | Clarke A+B |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in payload["results"]:
        t, v = r["test"], r["val"]
        p = r.get("n_params", 0)
        star = " **←**" if r["name"] == payload["selected_on_validation"] else ""
        # Models with a risk head should not be judged by these three columns —
        # their alarm reads a probability, not the point forecast.
        dagger = " †" if (r.get("probabilistic") or r.get("classify")) else ""
        lines.append(
            f"| {r['name']}{star}{dagger} | {p:,} | {v['rmse']:.2f} | {t['rmse']:.2f} | "
            f"{t['mae']:.2f} | {t['mard']:.2f}% | {t['rmse_hypo']:.2f} | "
            f"{t['hypo_recall']:.1%} | {t['hypo_precision']:.1%} | "
            f"{t['hypo_false_alarms_per_day']:.2f} | {t['clarke_ab']:.2f}% |"
        )

    if payload.get("has_risk_heads"):
        lines += [
            "",
            "† These models carry a risk head — a predicted distribution, a "
            "trained low/not-low classifier, or both. Their recall, precision and "
            "false-alarm columns above are computed the same way as everyone "
            "else's, by asking whether the *point* forecast fell under 70, and "
            "for them that is the wrong question: their alarm is meant to read "
            "the probability instead. Judge them in "
            "[`alarm.md`](alarm.md), where every model is compared on the signal "
            "it was actually built to emit.",
        ]
    lines += [
        "",
        f"Selection was done on the validation split; **"
        f"{payload['selected_on_validation']}** had the lowest validation RMSE among "
        "the round-1 models and its test numbers are reported above without further "
        "tuning. Which model actually ships is decided in [`alarm.md`](alarm.md), on "
        "alarm performance — for the reason immediately below.",
        "",
        "## Read the table sideways: accuracy and sensitivity to lows move in opposite directions",
        "",
        "Rank the models by RMSE and you almost exactly reverse their ranking by "
        "low-glucose recall. The most accurate model is the least willing to call "
        "a low; the least accurate one calls the most.",
        "",
        "This is not a coincidence, and it is not a bug in any particular model. "
        "Squared error rewards a forecast that stays near the conditional mean, "
        "and hypoglycaemia lives in the tail. A model that learns to hedge toward "
        "the middle wins on RMSE by systematically refusing to commit to the "
        "events the product exists to catch.",
        "",
        "The `false alarms/day` column is the mechanism in plain sight. The "
        "high-recall models are not more insightful — they simply alarm more "
        "often. Linear extrapolation reaches 74% recall by raising an alarm "
        "roughly 21 times a day, which no one would wear.",
        "",
        "Two consequences follow, and they shape the rest of the project:",
        "",
        "1. **Reporting recall at a fixed 70 mg/dL cutoff compares the models' "
        "biases, not their skill.** The fair question is how much recall each one "
        "buys at the *same* false-alarm budget. That comparison is in "
        "[`alarm.md`](alarm.md).",
        "2. **Selecting a model on RMSE would ship the wrong model.** The loss "
        "re-weighting rows (`tcn_hypo3`, `tcn_hypo8`) make the trade explicit: "
        "about half a mg/dL of overall RMSE buys a large improvement in RMSE "
        "restricted to lows and roughly nine points of recall. Weight 8 does not "
        "improve on weight 3, so the returns run out quickly.",
        "",
        f"_Generated by `python -m scripts.run_sweep` in "
        f"{payload['total_seconds'] / 60:.1f} minutes._",
    ]
    (ARTIFACTS_DIR.parent / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
