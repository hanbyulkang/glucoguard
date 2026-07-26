"""Score the four input variants as alarms, not as regressors.

RMSE already misled us once here: `treatments` improved validation RMSE and
then made test RMSE worse, while its low-glucose recall went *up*. That is the
same disagreement this project has hit repeatedly, accuracy and sensitivity to
lows are not the same axis, so the input experiment has to be settled on the
alarm, using the protocol the alarm pages use.

Three readings, in increasing order of how much they resemble wearing the thing:

1. recall at matched achieved false-alarm rates, pooled;
2. event-level episode recall with a shared threshold;
3. event-level episode recall with a per-wearer threshold.

Usage:  python -m scripts.eval_multimodal_alarm
"""
from __future__ import annotations

import json

import numpy as np
import torch

from src.alarm import pr_curve, risk_score, tune_threshold
from src.alarm_policy import event_metrics, tune_event_threshold
from src.calibration import split_by_time
from src.config import ARTIFACTS_DIR, HYPO_THRESHOLD
from src.data.windows_mm import aux_statistics, build_multimodal_windows
from src.models.nets import build as build_net
from src.train import pick_device

VARIANTS = ("cgm", "treatments", "devicestatus", "both")
RATES = (3.0, 8.0, 15.0)
TARGET_EVENTS = 6.0
WARMUP_DAYS = 14.0
MIN_WARMUP_LOWS = 20


def load_variant(name: str, aux_mean, aux_std, device):
    blob = torch.load(ARTIFACTS_DIR / f"mm_{name}.pt", map_location="cpu",
                      weights_only=False)
    norm = blob["norm"]
    model = build_net("tcn", mean=norm["mean"], std=norm["std"],
                      heteroscedastic=True, aux_mean=aux_mean, aux_std=aux_std)
    model.load_state_dict(blob["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def scores_of(model, X: np.ndarray, device, batch: int = 16_384) -> np.ndarray:
    """P(glucose < 70 in 30 min) for every window."""
    out = []
    for i in range(0, len(X), batch):
        chunk = torch.from_numpy(np.ascontiguousarray(X[i : i + batch])).to(device)
        out.append(model(chunk).float().cpu().numpy())
    pred = np.concatenate(out)
    return risk_score(pred[:, 0], pred[:, 1])


def recall_at(curve: dict, target: float) -> float:
    fa = np.asarray(curve["false_alarms_per_day"], dtype=float)
    rec = np.asarray(curve["recall"], dtype=float)
    order = np.argsort(fa)
    return float(np.interp(target, fa[order], rec[order]))


def per_wearer_events(ws, score: np.ndarray, shared: float) -> dict:
    """Episode recall under a shared cutoff and under each wearer's own."""
    pooled = {"shared": [], "personal": [], "y": []}
    fell_back = 0

    for pid in sorted(set(ws.patient_ids)):
        sel = ws.patient_ids == pid
        y, s = ws.y[sel], score[sel]
        t = np.asarray(ws.times[sel], dtype="datetime64[ns]")
        order = np.argsort(t)
        y, s, t = y[order], s[order], t[order]

        warm = split_by_time(t, WARMUP_DAYS)
        evaluation = ~warm
        if evaluation.sum() == 0:
            continue
        if int((y[warm] < HYPO_THRESHOLD).sum()) < MIN_WARMUP_LOWS:
            personal = shared
            fell_back += 1
        else:
            personal = tune_event_threshold(y[warm], s[warm], TARGET_EVENTS)

        pooled["y"].append(y[evaluation])
        pooled["shared"].append(s[evaluation] >= shared)
        pooled["personal"].append(s[evaluation] >= personal)

    y_all = np.concatenate(pooled["y"])
    out = {"fell_back": fell_back}
    for key in ("shared", "personal"):
        m = event_metrics(y_all, np.concatenate(pooled[key]))
        out[key] = {"episode_recall": m.episode_recall,
                    "false_alarms_per_day": m.false_alarms_per_day,
                    "episodes": m.episodes,
                    "median_lead_minutes": m.median_lead_minutes}
    return out


def main() -> None:
    device = pick_device()
    results = {}

    for name in VARIANTS:
        path = ARTIFACTS_DIR / f"mm_{name}.pt"
        if not path.exists():
            print(f"skipping {name}: no checkpoint")
            continue

        windows = build_multimodal_windows(name, verbose=False)
        train, val, test = windows["train"], windows["val"], windows["test"]
        aux_mean, aux_std = aux_statistics(train)
        model = load_variant(name, aux_mean, aux_std, device)

        s_val = scores_of(model, val.X, device)
        s_test = scores_of(model, test.X, device)

        curve = pr_curve(test.y, s_test, 400)
        shared = tune_event_threshold(val.y, s_val, TARGET_EVENTS)
        events = per_wearer_events(test, s_test, shared)

        results[name] = {
            "matched": {f"{r:g}": recall_at(curve, r) for r in RATES},
            "shared_threshold": shared,
            "events": events,
            "pr_curve_test": curve,
        }
        print(f"scored {name}", flush=True)
        del windows, model

    if not results:
        return

    print(f"\n{'=' * 96}\nPOOLED, recall at matched achieved false-alarm rates\n{'=' * 96}")
    print(f"{'inputs':<16}" + "".join(f"{f'{r:g} FA/day':>14s}" for r in RATES))
    base = results.get("cgm", {}).get("matched")
    for name, r in results.items():
        row = f"{name:<16}"
        for rate in RATES:
            v = r["matched"][f"{rate:g}"]
            delta = "" if not base else f" ({v - base[f'{rate:g}']:+.1%})"
            row += f"{v:>8.1%}{delta:>6s}"
        print(row)

    print(f"\n{'=' * 96}\nEVENT LEVEL, low episodes warned, on held-out wearers\n{'=' * 96}")
    print(f"{'inputs':<16}{'shared cutoff':>28s}{'per-wearer cutoff':>30s}")
    print(f"{'':<16}{'recall':>12s}{'FA/day':>9s}{'lead':>7s}"
          f"{'recall':>13s}{'FA/day':>9s}{'lead':>7s}")
    for name, r in results.items():
        sh, pe = r["events"]["shared"], r["events"]["personal"]
        print(f"{name:<16}{sh['episode_recall']:>12.1%}{sh['false_alarms_per_day']:>9.1f}"
              f"{sh['median_lead_minutes']:>6.0f}m"
              f"{pe['episode_recall']:>13.1%}{pe['false_alarms_per_day']:>9.1f}"
              f"{pe['median_lead_minutes']:>6.0f}m")

    payload = {"rates": list(RATES), "target_events": TARGET_EVENTS,
               "results": results}
    with open(ARTIFACTS_DIR / "multimodal_alarm.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=float)
    print(f"\nWrote {ARTIFACTS_DIR / 'multimodal_alarm.json'}")


if __name__ == "__main__":
    main()
