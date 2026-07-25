"""Sequence models for 30-minute glucose forecasting.

All three share the same contract so the training loop and the app can treat
them interchangeably: they take a raw window of glucose in mg/dL and return a
prediction in mg/dL.

Two choices apply to every model here:

* **They predict the change, not the level.** The network outputs how much
  glucose will move over the next 30 minutes and we add that to the current
  reading. Predicting the absolute level makes the network spend its capacity
  re-learning "the answer is near where we are now", which persistence already
  gives for free.
* **They see the rate of change explicitly.** The first difference of the window
  is handed over as a second channel. A network can derive it, but making it an
  input measurably shortens training and is what a clinician actually reads off
  a CGM trace.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _GlucoseFeatures(nn.Module):
    """Normalise a raw mg/dL window into (value, rate-of-change) channels."""

    def __init__(self, mean: float, std: float):
        super().__init__()
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T) in mg/dL  ->  (B, T, 2)
        value = (x - self.mean) / self.std
        delta = torch.diff(value, dim=1, prepend=value[:, :1])
        return torch.stack([value, delta], dim=-1)


class _DeltaHead(nn.Module):
    """Map a pooled representation to a glucose change in mg/dL.

    With ``heteroscedastic=True`` the head also emits a per-sample spread, so
    the model can report how confident it is rather than only what it expects.
    That matters more than it sounds: a point forecast trained on squared error
    is pulled toward the mean, which makes it systematically reluctant to
    predict the rare extremes — exactly the lows we care about. A predicted
    distribution lets the alarm ask "what is the probability of going below 70"
    instead of "did the single guessed number happen to land below 70".
    """

    def __init__(
        self, in_dim: int, hidden: int, std: float, dropout: float,
        heteroscedastic: bool = False,
    ):
        super().__init__()
        self.heteroscedastic = heteroscedastic
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2 if heteroscedastic else 1),
        )
        self.register_buffer("std", torch.tensor(float(std)))

    def forward(self, h: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        out = self.net(h)
        mu = last + out[..., 0] * self.std
        if not self.heteroscedastic:
            return mu
        # Softplus keeps the scale positive; the floor stops the NLL from
        # running away to zero variance on easy, flat stretches of the trace.
        sigma = nn.functional.softplus(out[..., 1]) * self.std + 1.0
        return torch.stack([mu, sigma], dim=-1)


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        mean: float,
        std: float,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.1,
        heteroscedastic: bool = False,
    ):
        super().__init__()
        self.features = _GlucoseFeatures(mean, std)
        self.rnn = nn.LSTM(
            input_size=2,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = _DeltaHead(hidden, hidden, std, dropout, heteroscedastic)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(self.features(x))
        return self.head(out[:, -1], x[:, -1])


class _TCNBlock(nn.Module):
    """Dilated causal convolution with a residual connection."""

    def __init__(self, channels: int, dilation: int, kernel: int, dropout: float):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(nn.functional.pad(x, (self.pad, 0)))))
        h = self.drop(h)
        h = self.act(self.norm2(self.conv2(nn.functional.pad(h, (self.pad, 0)))))
        return x + self.drop(h)


class TCNForecaster(nn.Module):
    def __init__(
        self,
        mean: float,
        std: float,
        channels: int = 96,
        levels: int = 4,
        kernel: int = 3,
        dropout: float = 0.1,
        heteroscedastic: bool = False,
    ):
        super().__init__()
        self.features = _GlucoseFeatures(mean, std)
        self.stem = nn.Conv1d(2, channels, 1)
        # Dilations double each level, so the receptive field covers the window.
        self.blocks = nn.Sequential(
            *[_TCNBlock(channels, 2**i, kernel, dropout) for i in range(levels)]
        )
        self.head = _DeltaHead(channels, channels, std, dropout, heteroscedastic)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).transpose(1, 2)     # (B, 2, T)
        h = self.blocks(self.stem(h))
        return self.head(h[:, :, -1], x[:, -1])


class TransformerForecaster(nn.Module):
    def __init__(
        self,
        mean: float,
        std: float,
        d_model: int = 96,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
        max_len: int = 64,
        heteroscedastic: bool = False,
    ):
        super().__init__()
        self.features = _GlucoseFeatures(mean, std)
        self.proj = nn.Linear(2, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = _DeltaHead(d_model, d_model * 2, std, dropout, heteroscedastic)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.features(x))
        h = h + self.pos[:, : h.size(1)]
        h = self.norm(self.encoder(h))
        return self.head(h[:, -1], x[:, -1])


REGISTRY = {
    "lstm": LSTMForecaster,
    "tcn": TCNForecaster,
    "transformer": TransformerForecaster,
}


def build(name: str, mean: float, std: float, **kwargs) -> nn.Module:
    if name not in REGISTRY:
        raise KeyError(f"unknown model '{name}'; choose from {sorted(REGISTRY)}")
    return REGISTRY[name](mean=mean, std=std, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
