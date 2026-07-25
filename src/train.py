"""Train one forecaster and score it on held-out patients.

Usage:
    python -m src.train --model lstm
    python -m src.train --model tcn --hypo-weight 4.0 --tag tcn_hypo
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from src.config import ARTIFACTS_DIR, HYPO_THRESHOLD
from src.data.windows import WindowSet, build_windows
from src.metrics import HEADER, evaluate, format_row
from src.models.nets import build, count_parameters


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    model: str = "lstm"
    epochs: int = 12
    batch_size: int = 2048
    steps_per_epoch: int = 1200
    lr: float = 2e-3
    weight_decay: float = 1e-4
    huber_delta: float = 10.0        # mg/dL; beyond this, errors count linearly
    hypo_weight: float = 0.0         # >0 upweights windows that end in a low
    probabilistic: bool = False      # predict a distribution, not a point
    patience: int = 3
    seed: int = 1337
    tag: str = ""


def sample_weights(y: torch.Tensor, strength: float) -> torch.Tensor | None:
    """Emphasise windows whose target is at or near hypoglycaemia.

    Lows are ~3% of windows, so an unweighted loss barely notices them. The ramp
    starts above the 70 mg/dL threshold on purpose: the clinically useful skill
    is seeing a low *coming*, which means getting the 70-100 band right too.
    """
    if strength <= 0:
        return None
    ramp = torch.sigmoid((HYPO_THRESHOLD + 30.0 - y) / 12.0)
    return 1.0 + strength * ramp


def gaussian_nll(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of a Gaussian predictive distribution.

    Optimising this rather than squared error lets the model widen its own
    error bars where the trace is genuinely unpredictable, instead of paying
    for that uncertainty by dragging the mean toward safety.
    """
    mu, sigma = out[..., 0], out[..., 1]
    return torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2


@torch.no_grad()
def predict(model: nn.Module, X: torch.Tensor, batch: int = 16384) -> np.ndarray:
    """Returns (n,) point forecasts, or (n, 2) of [mean, sigma] if probabilistic."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(model(X[i : i + batch]).float().cpu().numpy())
    return np.concatenate(out)


def as_point(pred: np.ndarray) -> np.ndarray:
    return pred[:, 0] if pred.ndim == 2 else pred


def run(cfg: TrainConfig, windows: dict[str, WindowSet] | None = None) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = pick_device()

    windows = windows or build_windows(verbose=False)
    train, val, test = windows["train"], windows["val"], windows["test"]

    mean, std = float(train.X.mean()), float(train.X.std())

    Xtr = torch.from_numpy(train.X).to(device)
    ytr = torch.from_numpy(train.y).to(device)
    Xva = torch.from_numpy(val.X).to(device)
    Xte = torch.from_numpy(test.X).to(device)

    model = build(cfg.model, mean=mean, std=std,
                  heteroscedastic=cfg.probabilistic).to(device)
    n_params = count_parameters(model)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * cfg.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=total_steps, pct_start=0.15
    )
    huber = nn.HuberLoss(delta=cfg.huber_delta, reduction="none")
    loss_fn = gaussian_nll if cfg.probabilistic else huber

    name = cfg.tag or cfg.model
    print(f"\n=== {name} | {n_params:,} params | device={device.type} ===", flush=True)

    best_rmse, best_state, bad_epochs = float("inf"), None, 0
    history = []
    n_train = len(ytr)
    started = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for _ in range(cfg.steps_per_epoch):
            idx = torch.randint(0, n_train, (cfg.batch_size,), device=device)
            xb, yb = Xtr[idx], ytr[idx]

            pred = model(xb)
            loss = loss_fn(pred, yb)
            w = sample_weights(yb, cfg.hypo_weight)
            loss = (loss * w).sum() / w.sum() if w is not None else loss.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()

        val_pred = predict(model, Xva)
        val_metrics = evaluate(val.y, as_point(val_pred))
        history.append({"epoch": epoch, "train_loss": running / cfg.steps_per_epoch,
                        **{k: val_metrics[k] for k in ("rmse", "mae", "hypo_recall")}})
        print(
            f"  epoch {epoch:2d}/{cfg.epochs}  loss {running / cfg.steps_per_epoch:7.3f}  "
            f"val RMSE {val_metrics['rmse']:6.2f}  MAE {val_metrics['mae']:6.2f}  "
            f"hypo recall {val_metrics['hypo_recall']:5.1%}  "
            f"[{time.time() - started:5.0f}s]",
            flush=True,
        )

        if val_metrics["rmse"] < best_rmse - 1e-4:
            best_rmse = val_metrics["rmse"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"  early stop at epoch {epoch} (no val gain for {cfg.patience})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_pred, test_pred = predict(model, Xva), predict(model, Xte)
    test_metrics = evaluate(test.y, as_point(test_pred))
    val_metrics = evaluate(val.y, as_point(val_pred))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": asdict(cfg),
         "norm": {"mean": mean, "std": std}, "n_params": n_params},
        ARTIFACTS_DIR / f"{name}.pt",
    )

    result = {
        "name": name, "model": cfg.model, "n_params": n_params,
        "config": asdict(cfg), "history": history,
        "val": val_metrics, "test": test_metrics,
        "train_seconds": round(time.time() - started, 1),
    }
    with open(ARTIFACTS_DIR / f"{name}.metrics.json", "w") as fh:
        json.dump({k: v for k, v in result.items() if not k.startswith("_")}, fh, indent=2)

    print(f"\n{HEADER}")
    print(format_row(f"{name} (test)", test_metrics), flush=True)

    # Kept out of the JSON (too large) but handed back so a caller can ensemble.
    result["_val_pred"] = val_pred
    result["_test_pred"] = test_pred
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="lstm", choices=["lstm", "tcn", "transformer"])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--steps-per-epoch", type=int, default=1200)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--hypo-weight", type=float, default=0.0)
    p.add_argument("--probabilistic", action="store_true")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    run(TrainConfig(
        model=args.model, epochs=args.epochs, batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch, lr=args.lr,
        hypo_weight=args.hypo_weight, probabilistic=args.probabilistic,
        tag=args.tag,
    ))


if __name__ == "__main__":
    main()
