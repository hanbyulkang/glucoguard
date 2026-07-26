"""Chart palette and Plotly styling.

Colours come from a validated categorical palette, the actual/predicted pair
was checked for colour-vision-deficiency separation and contrast against the
chart surface rather than picked by eye. Status colours are reserved for
clinical state (a low, a high) and never reused as a data series, so a red on
this screen always means the same thing.
"""
from __future__ import annotations

import plotly.graph_objects as go

# --- surfaces and ink ---------------------------------------------------------
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BORDER = "rgba(11,11,11,0.10)"

# --- categorical series (validated pair) --------------------------------------
ACTUAL = "#2a78d6"      # slot 1, blue
PREDICTED = "#eb6834"   # slot 2, orange

# --- reserved status ----------------------------------------------------------
CRITICAL = "#d03b3b"    # hypoglycaemia
WARNING = "#fab219"     # hyperglycaemia
GOOD = "#0ca30c"
MUTED_LINE = "#b5b4ae"   # de-emphasised series (baselines, context)

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def style(fig: go.Figure, height: int = 420, y_title: str = "", x_title: str = "") -> go.Figure:
    """Apply the shared chart chrome: recessive axes, hairline grid, unified hover."""
    fig.update_layout(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=13, color=INK_SECONDARY),
        margin=dict(l=8, r=76, t=8, b=8),   # room for the threshold labels
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=SURFACE, bordercolor=BORDER, font_size=13, font_family=FONT
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(color=INK_SECONDARY),
        ),
    )
    fig.update_xaxes(
        title_text=x_title,
        showgrid=False,
        zeroline=False,
        linecolor=AXIS,
        ticks="outside",
        tickcolor=AXIS,
        tickfont=dict(color=INK_MUTED, size=12),
    )
    fig.update_yaxes(
        title_text=y_title,
        title_font=dict(color=INK_MUTED, size=12),
        gridcolor=GRID,
        griddash="solid",
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickfont=dict(color=INK_MUTED, size=12),
    )
    return fig


CSS = f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1440px; }}
  #MainMenu, footer, header {{ visibility: hidden; }}

  .gg-title {{
    font-family: {FONT}; font-size: 1.9rem; font-weight: 650;
    color: {INK}; letter-spacing: -0.02em; margin: 0 0 .15rem 0;
  }}
  .gg-sub {{
    font-family: {FONT}; font-size: .98rem; color: {INK_SECONDARY};
    margin: 0 0 1.4rem 0; line-height: 1.5;
  }}
  .gg-banner {{
    font-family: {FONT}; display: flex; align-items: center; gap: .7rem;
    padding: .85rem 1.1rem; border-radius: 10px; margin: 0 0 1.1rem 0;
    font-size: .96rem; line-height: 1.45; border: 1px solid {BORDER};
  }}
  .gg-banner b {{ font-weight: 650; }}
  .gg-alert   {{ background: #fdeaea; color: #7d1d1d; border-color: rgba(208,59,59.35); }}
  .gg-caution {{ background: #fdf4e0; color: #6b4c07; border-color: rgba(250,178,25.40); }}
  .gg-ok      {{ background: #eaf6ea; color: #14550f; border-color: rgba(12,163,12.30); }}
  .gg-icon {{ font-size: 1.15rem; line-height: 1; }}

  .gg-tile {{
    border: 1px solid {BORDER}; border-radius: 10px; padding: .85rem 1rem;
    background: {SURFACE}; height: 100%;
  }}
  .gg-tile-label {{
    font-family: {FONT}; font-size: .78rem; text-transform: uppercase;
    letter-spacing: .06em; color: {INK_MUTED}; margin-bottom: .3rem;
  }}
  .gg-tile-value {{
    font-family: {FONT}; font-size: 1.55rem; font-weight: 650; color: {INK};
    line-height: 1.1;
  }}
  .gg-tile-note {{
    font-family: {FONT}; font-size: .8rem; color: {INK_SECONDARY}; margin-top: .25rem;
  }}
  .gg-caption {{
    font-family: {FONT}; font-size: .86rem; color: {INK_SECONDARY};
    margin: .1rem 0 1rem 0; line-height: 1.5;
  }}
  .gg-h2 {{
    font-family: {FONT}; font-size: 1.12rem; font-weight: 650; color: {INK};
    margin: 1.9rem 0 .35rem 0;
  }}
  .gg-h3 {{
    font-family: {FONT}; font-size: .95rem; font-weight: 650; color: {INK};
    margin: 1.3rem 0 .3rem 0;
  }}
  .gg-lead {{
    font-family: {FONT}; font-size: 1.02rem; color: {INK_SECONDARY};
    line-height: 1.62; margin: 0 0 1.1rem 0; max-width: 62ch;
  }}
  .gg-note {{
    font-family: {FONT}; font-size: .88rem; color: {INK_SECONDARY};
    line-height: 1.55; border-left: 3px solid {AXIS}; padding: .1rem 0 .1rem .8rem;
    margin: .9rem 0 1.2rem 0;
  }}
  .gg-note b {{ color: {INK}; }}
  .gg-pill {{
    display: inline-block; font-family: {FONT}; font-size: .74rem;
    letter-spacing: .05em; text-transform: uppercase; padding: .2rem .55rem;
    border-radius: 999px; border: 1px solid {BORDER}; color: {INK_MUTED};
    margin-right: .4rem;
  }}
  .gg-hero {{
    font-family: {FONT}; font-size: 2.6rem; font-weight: 680; color: {INK};
    line-height: 1; letter-spacing: -0.03em;
  }}
  .gg-hero-unit {{ font-size: 1rem; font-weight: 400; color: {INK_MUTED}; }}
  div[data-testid="stMetricValue"] {{ font-family: {FONT}; }}
</style>
"""
