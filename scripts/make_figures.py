"""Generate the two figures that carry the project's argument.

Figure 1 is the finding: across every model we trained, accuracy and
low-glucose sensitivity move in *opposite* directions. Optimising RMSE does not
merely fail to help the clinical objective — it actively works against it,
because squared error rewards a forecast that hugs the mean and lows are the
tail.

Figure 2 is the resolution: once each model's alarm threshold is tuned to a
common false-alarm budget, the apparent trade-off mostly dissolves, and what is
left is real skill.

Usage:  python -m scripts.make_figures
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import ARTIFACTS_DIR

ASSETS = ARTIFACTS_DIR.parent / "assets"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
CRITICAL = "#d03b3b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK_SECONDARY,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _clean(ax) -> None:
    ax.grid(axis="y", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, length=3)


def figure_tradeoff(sweep: dict) -> None:
    """RMSE against low-glucose recall, one dot per model."""
    rows = [r for r in sweep["results"] if not r["name"].startswith("ensemble")]
    x = [r["test"]["rmse"] for r in rows]
    y = [r["test"]["hypo_recall"] * 100 for r in rows]
    names = [r["name"] for r in rows]
    learned = [not n.startswith(("persistence", "linear", "ridge")) for n in names]

    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=200)
    for xi, yi, name, is_learned in zip(x, y, names, learned):
        colour = BLUE if is_learned else MUTED
        ax.scatter(xi, yi, s=110, color=colour, zorder=3,
                   edgecolor=SURFACE, linewidth=2)
        ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(9, 4),
                    fontsize=9, color=INK_SECONDARY)

    ax.set_xlabel("RMSE (mg/dL) — lower is better →", fontsize=10)
    ax.set_ylabel("Low-glucose recall (%) — higher is better ↑", fontsize=10)
    ax.set_title(
        "Every gain in accuracy cost sensitivity to lows",
        fontsize=12.5, color=INK, pad=12, loc="left", fontweight="semibold",
    )
    ax.text(
        0, 1.02,
        "Each model, scored on held-out patients at a fixed 70 mg/dL alarm cutoff",
        transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom",
    )
    _clean(ax)
    ax.grid(axis="x", linewidth=0.8, alpha=0.9)
    fig.tight_layout()
    fig.savefig(ASSETS / "tradeoff.png", facecolor=SURFACE)
    plt.close(fig)


def figure_alarm(alarm: dict) -> None:
    """Recall against false alarms per day — the fair comparison."""
    keep = [n for n in ("persistence", "ridge", "linear_extrapolation") if n in alarm]
    learned = [n for n in alarm if n not in keep]
    # Show the best learned alarm plus the baselines, not all fifteen curves.
    learned.sort(key=lambda n: -alarm[n]["budgets"][
        list(alarm[n]["budgets"])[-1]]["recall"])
    show = keep + learned[:2]

    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=200)
    palette = {n: c for n, c in zip(show, [MUTED, "#8a8a86", "#b5b4ae", BLUE, ORANGE])}
    for name in show:
        curve = alarm[name]["pr_curve_test"]
        fa, rec = curve["false_alarms_per_day"], [r * 100 for r in curve["recall"]]
        pairs = [(f, r) for f, r in zip(fa, rec) if f <= 24]
        if not pairs:
            continue
        is_learned = name in learned
        ax.plot([p[0] for p in pairs], [p[1] for p in pairs],
                color=palette.get(name, MUTED),
                linewidth=2.4 if is_learned else 1.6,
                label=name, zorder=3 if is_learned else 2)

    ax.set_xlabel("False alarms per day (lower is better) →", fontsize=10)
    ax.set_ylabel("Low-glucose recall (%) ↑", fontsize=10)
    ax.set_title(
        "Compared at the same false-alarm budget, the trade-off dissolves",
        fontsize=12.5, color=INK, pad=12, loc="left", fontweight="semibold",
    )
    ax.text(
        0, 1.02,
        "Alarm threshold swept across its full range, on held-out patients",
        transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower right")
    _clean(ax)
    fig.tight_layout()
    fig.savefig(ASSETS / "alarm_curve.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    sweep_path, alarm_path = ARTIFACTS_DIR / "sweep.json", ARTIFACTS_DIR / "alarm.json"

    if sweep_path.exists():
        figure_tradeoff(json.loads(sweep_path.read_text()))
        print(f"wrote {ASSETS / 'tradeoff.png'}")
    if alarm_path.exists():
        figure_alarm(json.loads(alarm_path.read_text()))
        print(f"wrote {ASSETS / 'alarm_curve.png'}")


if __name__ == "__main__":
    main()
