"""Fine-tune the shared model on one wearer's own history.

Recalibrating the threshold adapts *when* the system speaks. It cannot change
what the network believes, and the within-wearer analysis left a small residual
drift — around 1 mg/dL by the second year — that no threshold can touch. If the
person or their hardware has moved away from what the network learned, the
network has to move too.

The rules that make this honest are the same ones that governed everything else:

* **Strictly causal.** A model applied on a given day is fine-tuned only on that
  wearer's readings from strictly before it. Nothing from the evaluation period
  ever reaches the weights.
* **Always restart from the shared model.** Each refresh fine-tunes the original
  global weights on the trailing window rather than continuing from the previous
  personal copy. Chaining would let one bad window compound quietly across years.
* **A small budget on purpose.** A few hundred steps at a low learning rate on a
  month of one person's data. Anything more and the network stops being a
  glucose model that knows this wearer, and becomes a model of one month.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.config import HYPO_THRESHOLD
from src.train import gaussian_nll

DEFAULT_STEPS = 300
DEFAULT_LR = 1e-4
DEFAULT_BATCH = 256
MIN_WINDOWS = 3_000       # roughly ten days of wear
MIN_LOWS = 40             # without lows there is nothing personal to learn


BATCH_PREDICT = 16_384


def batched_predict(model: nn.Module, X: np.ndarray, device: torch.device,
                    batch: int = BATCH_PREDICT) -> np.ndarray:
    """Predict in chunks. A wearer's full record is up to 400k windows, and
    pushing that through the GPU in one call exhausts unified memory."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            chunk = torch.from_numpy(np.ascontiguousarray(X[i : i + batch])).to(device)
            out.append(model(chunk).float().cpu().numpy())
    return np.concatenate(out)


@dataclass
class FineTuneReport:
    steps: int
    windows: int
    lows: int
    loss_before: float
    loss_after: float
    skipped: str = ""


def _loss_of(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
             probabilistic: bool, huber: nn.Module) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(X), BATCH_PREDICT):
            xb, yb = X[i : i + BATCH_PREDICT], y[i : i + BATCH_PREDICT]
            out = model(xb)
            if probabilistic:
                loss = gaussian_nll(out[..., :2], yb)
            else:
                mu = out[..., 0] if out.ndim == 2 else out
                loss = huber(mu, yb)
            total += float(loss.sum())
            n += len(yb)
    return total / max(n, 1)


def fine_tune(base: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device,
              probabilistic: bool, classify: bool = False,
              steps: int = DEFAULT_STEPS, lr: float = DEFAULT_LR,
              batch: int = DEFAULT_BATCH) -> tuple[nn.Module | None, FineTuneReport]:
    """Return a personal copy of `base`, or None when the window is too thin."""
    lows = int((y < HYPO_THRESHOLD).sum())
    if len(y) < MIN_WINDOWS or lows < MIN_LOWS:
        return None, FineTuneReport(0, len(y), lows, float("nan"), float("nan"),
                                    skipped="not enough personal history")

    model = copy.deepcopy(base).to(device)
    Xt = torch.from_numpy(np.ascontiguousarray(X)).to(device)
    yt = torch.from_numpy(np.ascontiguousarray(y)).to(device)

    huber = nn.HuberLoss(delta=10.0, reduction="none")
    bce = nn.BCEWithLogitsLoss(reduction="none")
    before = _loss_of(model, Xt, yt, probabilistic, huber)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    n = len(yt)
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), device=device)
        xb, yb = Xt[idx], yt[idx]
        out = model(xb)
        if probabilistic:
            loss = gaussian_nll(out[..., :2], yb)
        else:
            mu = out[..., 0] if out.ndim == 2 else out
            loss = huber(mu, yb)
        if classify:
            loss = loss + bce(out[..., -1], (yb < HYPO_THRESHOLD).float())
        loss = loss.mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    after = _loss_of(model, Xt, yt, probabilistic, huber)
    model.eval()
    return model, FineTuneReport(steps, len(y), lows, before, after)


def rolling_personal_predictions(
    base: nn.Module, X: np.ndarray, y: np.ndarray, times: np.ndarray,
    device: torch.device, probabilistic: bool, classify: bool = False,
    trailing_days: float = 90.0, refresh_days: float = 90.0,
    steps: int = DEFAULT_STEPS, lr: float = DEFAULT_LR,
) -> tuple[np.ndarray, list[FineTuneReport]]:
    """Predictions where each block is produced by a model refreshed before it.

    Returns an array shaped like the base model's output, with NaN wherever no
    personal model was available yet — the caller falls back to the shared model
    there rather than pretending coverage it does not have.
    """
    t = np.asarray(times, dtype="datetime64[ns]")
    order = np.argsort(t)
    t_s, X_s, y_s = t[order], X[order], y[order]

    with torch.no_grad():
        probe = base(torch.from_numpy(X[:1]).to(device))
    width = probe.shape[-1] if probe.ndim == 2 else 1
    out = np.full((len(y), width) if width > 1 else (len(y),), np.nan, dtype=np.float32)

    trailing = np.timedelta64(int(trailing_days * 24 * 60), "m")
    refresh = np.timedelta64(int(refresh_days * 24 * 60), "m")
    edge = t_s[0] + trailing
    reports: list[FineTuneReport] = []

    while edge <= t_s[-1]:
        window = (t_s >= edge - trailing) & (t_s < edge)
        applies = (t_s >= edge) & (t_s < edge + refresh)
        if applies.sum() and window.sum():
            model, report = fine_tune(base, X_s[window], y_s[window], device,
                                      probabilistic, classify, steps, lr)
            reports.append(report)
            if model is not None:
                out[order[applies]] = batched_predict(model, X_s[applies], device)
        edge = edge + refresh

    return out, reports
