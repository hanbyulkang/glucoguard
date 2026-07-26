"""Four experiments: CGM alone, plus what the wearer did, plus what the loop knew.

The CGM-only model is blind at the moments a person acts. This measures how much
of that blindness the archive can fill, and separates two very different kinds of
extra information:

* **treatments**, boluses, carbohydrates, basal rates. What the wearer actually
  did, hand-entered and therefore incomplete.
* **devicestatus**, insulin on board, carbs on board, insulin activity. What
  OpenAPS computed, already convolved through its own pharmacokinetic model.

Everything else is held fixed: same architecture, same patient split, same
training budget, same seed. Only the input channels change.

Usage:  python -m scripts.run_multimodal
"""
from __future__ import annotations

import json

import numpy as np
import torch

from src.config import ARTIFACTS_DIR
from src.data.windows_mm import FEATURE_SETS, aux_statistics, build_multimodal_windows
from src.metrics import HEADER, evaluate, format_row
from src.models.nets import build as build_net, count_parameters
from src.train import TrainConfig, gaussian_nll, pick_device, predict

ARCH = "tcn"
EPOCHS = 12
STEPS = 1200
BATCH = 2048
LR = 2e-3
SEED = 1337


def train_one(name: str, windows: dict, device) -> dict:
    train, val, test = windows["train"], windows["val"], windows["test"]
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    glucose = train.X[... 0] if train.X.ndim == 3 else train.X
    mean, std = float(glucose.mean()), float(glucose.std())
    aux_mean, aux_std = aux_statistics(train)

    model = build_net(ARCH, mean=mean, std=std, heteroscedastic=True,
                      aux_mean=aux_mean, aux_std=aux_std).to(device)
    n_params = count_parameters(model)

    Xtr = torch.from_numpy(train.X).to(device)
    ytr = torch.from_numpy(train.y).to(device)
    Xva = torch.from_numpy(val.X).to(device)
    Xte = torch.from_numpy(test.X).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=EPOCHS * STEPS, pct_start=0.15)

    n = len(ytr)
    best, best_state, bad = float("inf"), None, 0
    print(f"\n=== {name}: input {train.X.shape[1:]}, {n_params:,} params ===",
          flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for _ in range(STEPS):
            idx = torch.randint(0, n, (BATCH,), device=device)
            loss = gaussian_nll(model(Xtr[idx])[... :2], ytr[idx]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()

        m = evaluate(val.y, predict(model, Xva)[... 0])
        print(f"  epoch {epoch:2d}  val RMSE {m['rmse']:6.2f}", flush=True)
        if m["rmse"] < best - 1e-4:
            best, bad = m["rmse"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                print(f"  early stop at epoch {epoch}", flush=True)
                break

    if best_state:
        model.load_state_dict(best_state)

    val_m = evaluate(val.y, predict(model, Xva)[... 0])
    test_m = evaluate(test.y, predict(model, Xte)[... 0])
    torch.save({"state_dict": model.state_dict(),
                "config": {"model": ARCH, "probabilistic": True, "classify": False,
                           "feature_set": name},
                "norm": {"mean": mean, "std": std},
                "aux": {"mean": None if aux_mean is None else aux_mean.tolist(),
                        "std": None if aux_std is None else aux_std.tolist()},
                "n_params": n_params},
               ARTIFACTS_DIR / f"mm_{name}.pt")
    return {"name": name, "n_params": n_params,
            "channels": list(train.X.shape[1:]), "val": val_m, "test": test_m}


def main() -> None:
    device = pick_device()
    results = []
    for name in ("cgm", "treatments", "devicestatus", "both"):
        windows = build_multimodal_windows(name, verbose=True)
        results.append(train_one(name, windows, device))
        del windows

    print(f"\n{'=' * 100}\nMULTIMODAL COMPARISON (test split)\n{'=' * 100}")
    print(f"{'inputs':<16}{HEADER}")
    base = next(r for r in results if r["name"] == "cgm")["test"]["rmse"]
    for r in results:
        delta = r["test"]["rmse"] - base
        print(f"{r['name']:<16}" + format_row("", r["test"]) + f"   {delta:+6.2f}")

    with open(ARTIFACTS_DIR / "multimodal.json", "w") as fh:
        json.dump({"arch": ARCH, "feature_sets": FEATURE_SETS,
                   "results": results}, fh, indent=2)
    print(f"\nWrote {ARTIFACTS_DIR / 'multimodal.json'}")


if __name__ == "__main__":
    main()
