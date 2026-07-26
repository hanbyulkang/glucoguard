"""Non-learned and shallow baselines.

Every number a neural network produces is meaningless without these. Persistence
in particular is the honest floor: CGM at a 30-minute horizon is so
autocorrelated that "assume nothing changes" is already a decent forecast, and
papers that omit it can make a mediocre model look impressive.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from src.config import HORIZON_STEPS, SAMPLE_MINUTES


def persistence(X: np.ndarray) -> np.ndarray:
    """Predict that glucose 30 minutes from now equals glucose right now."""
    return X[:, -1].copy()


def linear_extrapolation(X: np.ndarray, fit_steps: int = 6) -> np.ndarray:
    """Fit a straight line to the last `fit_steps` samples and extend it.

    Deliberately included because the MetaboNet-Bench authors found that plain
    linear extrapolation beats far heavier models specifically in the low-glucose
    region, the region we care most about. If our network cannot beat this
    there, the network is not earning its complexity.
    """
    recent = X[:, -fit_steps:]
    t = np.arange(fit_steps, dtype=np.float64)
    t_centred = t - t.mean()
    denom = np.sum(t_centred**2)

    slope = (recent * t_centred).sum(axis=1) / denom     # mg/dL per 5-min step
    intercept = recent.mean(axis=1)
    steps_ahead = (fit_steps - 1) / 2 + HORIZON_STEPS
    return intercept + slope * steps_ahead


class RidgeBaseline:
    """Ridge regression on the raw 2-hour history vector.

    A linear model with access to the full window. It sets the bar that any
    recurrent or attention-based model has to clear to justify itself.
    """

    def __init__(self, alpha: float = 1.0, predict_delta: bool = True):
        self.model = Ridge(alpha=alpha)
        self.predict_delta = predict_delta

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeBaseline":
        target = y - X[:, -1] if self.predict_delta else y
        self.model.fit(X, target)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = self.model.predict(X)
        return out + X[:, -1] if self.predict_delta else out


BASELINES = {
    "persistence": persistence,
    f"linear_extrap_{SAMPLE_MINUTES * 6}min": linear_extrapolation,
}
