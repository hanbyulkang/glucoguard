"""The patient-facing face of GlucoGuard.

Everything else in this app is evidence for a reviewer: RMSE, Clarke zones,
false-alarm budgets. None of that belongs in front of the person wearing the
sensor. What they need is one number, one direction, and one sentence about
whether they have to do something in the next half hour.

So this module renders a phone, and the phone shows almost nothing.
"""
from __future__ import annotations

from src.theme import FONT

# Status palette, kept deliberately distinct from the chart series colours so a
# red here never reads as "series 3".
STATES = {
    "low_now": dict(bg="#7d1d1d", fg="#ffffff", accent="#ff8b8b",
                    label="Low now"),
    "low_soon": dict(bg="#b8332f", fg="#ffffff", accent="#ffd0cd",
                     label="Going low"),
    "watch": dict(bg="#8a6100", fg="#ffffff", accent="#ffe4a3",
                  label="Drifting down"),
    "ok": dict(bg="#14312a", fg="#ffffff", accent="#8fe0c4",
               label="In range"),
    "high": dict(bg="#7a5a12", fg="#ffffff", accent="#ffd98a",
                 label="Above range"),
    "unknown": dict(bg="#3a3a38", fg="#ffffff", accent="#c3c2b7",
                    label="No reading"),
}

ARROWS = {
    "rising_fast": "↑↑", "rising": "↑", "flat": "→",
    "falling": "↓", "falling_fast": "↓↓",
}


def trend_arrow(delta_per_5min: float) -> tuple[str, str]:
    """CGM-style trend arrow from the recent rate of change, in mg/dL per 5 min."""
    rate = delta_per_5min * 3          # convert to mg/dL per 15 min, the usual basis
    if rate >= 15:
        return ARROWS["rising_fast"], "rising quickly"
    if rate >= 5:
        return ARROWS["rising"], "rising"
    if rate <= -15:
        return ARROWS["falling_fast"], "falling quickly"
    if rate <= -5:
        return ARROWS["falling"], "falling"
    return ARROWS["flat"], "steady"


def classify(now: float | None, predicted: float | None, risk: float | None,
             threshold: float | None) -> str:
    if now is None:
        return "unknown"
    if now < 70:
        return "low_now"
    if risk is not None and threshold is not None and risk >= threshold:
        return "low_soon"
    if predicted is not None and predicted < 85:
        return "watch"
    if now > 180:
        return "high"
    return "ok"


def headline(state: str, now: float, predicted: float | None,
             minutes: int, risk: float | None) -> tuple[str, str]:
    """The one line the wearer reads, and the smaller line under it."""
    if state == "unknown":
        return ("No recent reading",
                "The sensor has not reported for a while, so there is no forecast.")
    if state == "low_now":
        return (f"You are low, {now:.0f}",
                "Treat this now. The forecast below is what happens next.")
    if state == "low_soon":
        chance = f" ({risk:.0%} chance)" if risk is not None else ""
        return (f"Heading low in about {minutes} minutes{chance}",
                f"Predicted {predicted:.0f} mg/dL. You have time to act before it happens.")
    if state == "watch":
        return (f"Drifting down, {predicted:.0f} expected",
                f"Not a low yet, but worth watching over the next {minutes} minutes.")
    if state == "high":
        return (f"Above range, {now:.0f}",
                f"Predicted {predicted:.0f} mg/dL in {minutes} minutes.")
    return (f"In range, {now:.0f}",
            f"Predicted {predicted:.0f} mg/dL in {minutes} minutes. Nothing to do.")


def _flatten(html: str) -> str:
    """Collapse indentation before handing HTML to Streamlit.

    `st.markdown` runs the string through a Markdown parser first, and Markdown
    treats any line indented four spaces or more as a code block, so nicely
    formatted HTML arrives on screen as literal `<div style=...>` text.
    """
    return " ".join(line.strip() for line in html.splitlines() if line.strip())


def phone(now: float | None, predicted: float | None, risk: float | None,
          threshold: float | None, minutes: int, delta_per_5min: float,
          clock: str, spark: list[float], notification: bool) -> str:
    """Render the whole phone as one HTML string."""
    state = classify(now, predicted, risk, threshold)
    s = STATES[state]
    arrow, trend_word = trend_arrow(delta_per_5min)
    title, sub = (headline(state, now, predicted, minutes, risk)
                  if now is not None else headline("unknown", 0, None, minutes, None))

    spark_svg = _sparkline(spark, s["accent"]) if spark else ""
    risk_bar = ""
    if risk is not None:
        pct = max(2.0, min(100.0, risk * 100))
        marker = ""
        if threshold is not None:
            marker = (f'<div style="position:absolute;left:{min(98, threshold * 100):.1f}%;'
                      f'top:-4px;width:2px;height:18px;background:{s["fg"]};opacity:.55"></div>')
        risk_bar = f"""
        <div style="margin-top:20px">
          <div style="display:flex;justify-content:space-between;font-size:11px;
                      letter-spacing:.06em;text-transform:uppercase;opacity:.65">
            <span>chance of going low</span><span>{risk:.0%}</span>
          </div>
          <div style="position:relative;margin-top:7px;height:10px;border-radius:99px;
                      background:rgba(255,255,255.16)">
            <div style="width:{pct:.1f}%;height:100%;border-radius:99px;
                        background:{s['accent']}"></div>{marker}
          </div>
          <div style="font-size:10.5px;opacity:.55;margin-top:5px">
            the mark is your own alarm setting
          </div>
        </div>"""

    banner = ""
    if notification:
        banner = f"""
        <div style="position:absolute;top:14px;left:14px;right:14px;z-index:5;
                    background:rgba(255,255,255.96);border-radius:16px;padding:11px 13px;
                    box-shadow:0 8px 24px rgba(0,0,0.28);font-family:{FONT};
                    text-align:left">
          <div style="font-size:10.5px;color:#898781;letter-spacing:.04em;
                      text-transform:uppercase">GlucoGuard · now</div>
          <div style="font-size:13.5px;color:#0b0b0b;font-weight:640;margin-top:3px">
            Glucose heading low</div>
          <div style="font-size:12.5px;color:#52514e;margin-top:2px;line-height:1.35">
            {predicted:.0f} mg/dL expected in {minutes} min. Not a dose recommendation.
          </div>
        </div>"""

    now_text = f"{now:.0f}" if now is not None else "--"

    return _flatten(f"""
    <div style="display:flex;justify-content:center;padding:6px 0 18px 0">
      <div style="position:relative;width:330px;border-radius:44px;padding:11px;
                  background:#111110;box-shadow:0 26px 60px rgba(0,0,0.30)">
        <div style="border-radius:34px;overflow:hidden;background:{s['bg']};
                    color:{s['fg']};font-family:{FONT};position:relative;
                    min-height:600px">
          {banner}
          <div style="padding:26px 24px 22px 24px">
            <div style="display:flex;justify-content:space-between;align-items:center;
                        font-size:12px;opacity:.7">
              <span>{clock}</span>
              <span style="letter-spacing:.07em;text-transform:uppercase">{s['label']}</span>
            </div>

            <div style="margin-top:34px;display:flex;align-items:baseline;gap:12px">
              <div style="font-size:86px;font-weight:300;line-height:.86;
                          letter-spacing:-.045em">{now_text}</div>
              <div style="font-size:34px;opacity:.85;line-height:1">{arrow}</div>
            </div>
            <div style="font-size:13px;opacity:.7;margin-top:6px">
              mg/dL · {trend_word}
            </div>

            {spark_svg}

            <div style="margin-top:24px;font-size:20.5px;font-weight:620;
                        line-height:1.28;letter-spacing:-.01em">{title}</div>
            <div style="margin-top:8px;font-size:13.5px;opacity:.78;line-height:1.5">
              {sub}</div>

            {risk_bar}

            <div style="margin-top:26px;padding-top:14px;
                        border-top:1px solid rgba(255,255,255.14);
                        font-size:11px;opacity:.55;line-height:1.5">
              GlucoGuard does not recommend insulin doses. Research demonstration,
              not a medical device.
            </div>
          </div>
        </div>
      </div>
    </div>""")


def _sparkline(values: list[float], colour: str, width: int = 282,
               height: int = 74) -> str:
    """Last couple of hours as a bare line: no axes, no numbers."""
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = max(hi - lo, 25.0)
    mid = (hi + lo) / 2
    lo, hi = mid - span / 2, mid + span / 2

    step = width / (len(values) - 1)
    pts = " ".join(
        f"{i * step:.1f},{height - (v - lo) / (hi - lo) * height:.1f}"
        for i, v in enumerate(values)
    )
    y70 = height - (70 - lo) / (hi - lo) * height
    threshold_line = ""
    if 0 <= y70 <= height:
        threshold_line = (f'<line x1="0" y1="{y70:.1f}" x2="{width}" y2="{y70:.1f}" '
                          f'stroke="rgba(255,255,255.28)" stroke-width="1" '
                          f'stroke-dasharray="3 4"/>')
    last_x = (len(values) - 1) * step
    last_y = height - (values[-1] - lo) / (hi - lo) * height
    return _flatten(f"""
    <div style="margin-top:22px">
      <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
        {threshold_line}
        <polyline points="{pts}" fill="none" stroke="{colour}"
                  stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>
        <circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4.5" fill="{colour}"/>
      </svg>
      <div style="font-size:10.5px;opacity:.5;margin-top:2px">
        last 2 hours · dashed line is 70 mg/dL
      </div>
    </div>""")
