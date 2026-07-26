"""The part of GlucoGuard that runs by itself.

A demo that shows a forecast when you press a button is a picture of a product.
The thing that makes it a product is that nobody presses anything: it reads the
sensor on a timer, decides whether to speak, and stays quiet when it has already
said the same thing five minutes ago.

That last part is most of the work. A naive implementation alarms on every
reading while glucose stays low, which is how alarm fatigue is manufactured.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from src.config import HORIZON_MINUTES, HYPO_THRESHOLD

SNOOZE_MINUTES = 30
ESCALATE_MINUTES = 60      # if still heading low this long after the last alert


@dataclass
class AlertRecord:
    at: datetime
    glucose: float
    predicted: float
    risk: float
    delivered: str          # what the notifier reported back
    kind: str               # "warning" | "still low" | "recovered"


@dataclass
class MonitorState:
    """Everything the loop needs to remember between ticks."""

    last_alert_at: datetime | None = None
    last_state: str = "ok"
    log: list[AlertRecord] = field(default_factory=list)
    ticks: int = 0

    def to_dict(self) -> dict:
        return {"last_alert_at": self.last_alert_at, "last_state": self.last_state,
                "log": self.log, "ticks": self.ticks}


def should_alert(state: MonitorState, alarming: bool, now: datetime,
                 glucose: float) -> tuple[bool, str]:
    """Decide whether this tick is worth interrupting someone over.

    Three rules, in order:

    * Nothing to say when the risk is below the wearer's own cutoff.
    * Say it once, then stay quiet for the snooze window — a low that lasts an
      hour is one event, not twelve.
    * Speak again if it is *still* going after the escalation window, because
      silence at that point is indistinguishable from the app having crashed.
    """
    if not alarming:
        # Recovery is worth one quiet note, but only if we had raised something.
        if state.last_state in ("warning", "still low") and glucose >= HYPO_THRESHOLD:
            return True, "recovered"
        return False, ""

    if state.last_alert_at is None:
        return True, "warning"

    since = (now - state.last_alert_at).total_seconds() / 60
    if since >= ESCALATE_MINUTES:
        return True, "still low"
    if since >= SNOOZE_MINUTES and state.last_state == "recovered":
        return True, "warning"
    return False, ""


def alert_text(kind: str, glucose: float, predicted: float,
               risk: float) -> tuple[str, str]:
    """Title and body for the push, in a wearer's language."""
    if kind == "recovered":
        return ("Back in range",
                f"Glucose is {glucose:.0f} mg/dL. No action needed.")
    if kind == "still low":
        return ("Still heading low",
                f"{glucose:.0f} mg/dL now, {predicted:.0f} expected in "
                f"{HORIZON_MINUTES} minutes. This is a repeat of an earlier "
                f"warning.")
    return (f"Glucose heading low — {HORIZON_MINUTES} min warning",
            f"{glucose:.0f} mg/dL now, {predicted:.0f} mg/dL expected in "
            f"{HORIZON_MINUTES} minutes ({risk:.0%} chance of going under "
            f"{HYPO_THRESHOLD}). Research demo, not a medical device — do not "
            f"treat based on this.")


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def summarise(log: list[AlertRecord], hours: float = 24.0) -> dict:
    """What the wearer would say if asked how noisy the last day was."""
    if not log:
        return {"alerts": 0, "warnings": 0, "per_day": 0.0, "last": None}
    cutoff = log[-1].at - timedelta(hours=hours)
    recent = [a for a in log if a.at >= cutoff]
    warnings = [a for a in recent if a.kind != "recovered"]
    span_hours = max(
        (recent[-1].at - recent[0].at).total_seconds() / 3600, 1e-6
    ) if len(recent) > 1 else hours
    return {
        "alerts": len(recent),
        "warnings": len(warnings),
        "per_day": len(warnings) / max(span_hours / 24, 1e-6),
        "last": log[-1],
    }
