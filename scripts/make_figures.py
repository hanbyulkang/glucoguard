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


def _place_labels(ax, xs, ys, names, fontsize: float = 9) -> None:
    """Annotate points, nudging each label to the first spot that is free.

    Several of these models land almost on top of each other — that proximity
    is itself part of the finding — so fixed offsets produce an unreadable pile.
    Candidate positions are tried in order of preference and the first one that
    collides with neither a placed label nor a data point wins.
    """
    fig = ax.figure
    fig.canvas.draw()                      # need a renderer for display coords
    px = ax.transData.transform(list(zip(xs, ys)))

    char_w, line_h = fontsize * 0.70, fontsize * 1.7
    candidates = [(13, 4), (13, -12), (-13, 4), (-13, -12),
                  (13, 16), (-13, 16), (13, -24), (-13, -24),
                  (0, 20), (0, -28)]

    placed: list[tuple[float, float, float, float]] = []
    point_boxes = [(x - 7, y - 7, x + 7, y + 7) for x, y in px]

    def overlaps(box, others) -> bool:
        return any(not (box[2] < o[0] or box[0] > o[2]
                        or box[3] < o[1] or box[1] > o[3]) for o in others)

    order = sorted(range(len(names)), key=lambda i: -px[i][1])
    for i in order:
        x, y = px[i]
        w, h = len(names[i]) * char_w + 6, line_h
        for dx, dy in candidates:
            left = x + dx if dx >= 0 else x + dx - w
            box = (left, y + dy - h * 0.25, left + w, y + dy + h * 0.75)
            if not overlaps(box, placed + point_boxes):
                placed.append(box)
                ax.annotate(names[i], (xs[i], ys[i]), textcoords="offset points",
                            xytext=(dx, dy), fontsize=fontsize,
                            color=INK_SECONDARY,
                            ha="left" if dx >= 0 else "right")
                break
        else:
            ax.annotate(names[i], (xs[i], ys[i]), textcoords="offset points",
                        xytext=(13, 4), fontsize=fontsize, color=INK_SECONDARY)


def figure_tradeoff(sweep: dict) -> None:
    """RMSE against low-glucose recall, one dot per model."""
    rows = [r for r in sweep["results"] if not r["name"].startswith("ensemble")]
    x = [r["test"]["rmse"] for r in rows]
    y = [r["test"]["hypo_recall"] * 100 for r in rows]
    names = [r["name"] for r in rows]
    learned = [not n.startswith(("persistence", "linear", "ridge")) for n in names]

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=200)
    ax.scatter(x, y, s=115, zorder=3, edgecolor=SURFACE, linewidth=2,
               color=[BLUE if L else MUTED for L in learned])

    ax.set_xlabel("RMSE (mg/dL), lower is better", fontsize=10)
    ax.set_ylabel("Low-glucose recall (%), higher is better", fontsize=10)
    ax.set_xlim(min(x) - 2.0, max(x) + 2.4)
    ax.set_ylim(min(y) - 8, max(y) + 9)
    _place_labels(ax, x, y, names)

    fig.suptitle(
        "Every gain in accuracy cost sensitivity to lows",
        fontsize=13, color=INK, fontweight="bold", x=0.012, ha="left", y=0.985,
    )
    ax.set_title(
        "Each model on held-out patients, alarm read at a fixed 70 mg/dL cutoff",
        fontsize=9.5, color=MUTED, loc="left", pad=10,
    )
    _clean(ax)
    ax.grid(axis="x", linewidth=0.8, alpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
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

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=200)
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

    ax.set_xlabel("False alarms per day, lower is better", fontsize=10)
    ax.set_ylabel("Low-glucose recall (%)", fontsize=10)
    fig.suptitle(
        "Compared at the same false-alarm budget, the trade-off dissolves",
        fontsize=13, color=INK, fontweight="bold", x=0.012, ha="left", y=0.985,
    )
    ax.set_title(
        "Alarm threshold swept across its full range, on held-out patients",
        fontsize=9.5, color=MUTED, loc="left", pad=10,
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="lower right")
    _clean(ax)
    ax.grid(axis="x", linewidth=0.8, alpha=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
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
