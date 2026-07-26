"""Small building blocks shared by every page of the app."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.theme import CSS, INK_MUTED


def page(title: str, subtitle: str = "", pills: list[str] | None = None) -> None:
    """Standard page header. Call once, first thing, on every page."""
    st.markdown(CSS, unsafe_allow_html=True)
    if pills:
        st.markdown("".join(f'<span class="gg-pill">{p}</span>' for p in pills),
                    unsafe_allow_html=True)
    st.markdown(f'<div class="gg-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="gg-lead">{subtitle}</div>', unsafe_allow_html=True)


def h2(text: str) -> None:
    st.markdown(f'<div class="gg-h2">{text}</div>', unsafe_allow_html=True)


def h3(text: str) -> None:
    st.markdown(f'<div class="gg-h3">{text}</div>', unsafe_allow_html=True)


def caption(text: str) -> None:
    st.markdown(f'<div class="gg-caption">{text}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    """A quieter aside: caveats, definitions, things that qualify a number."""
    st.markdown(f'<div class="gg-note">{text}</div>', unsafe_allow_html=True)


def tile(label: str, value: str, note_text: str = "") -> str:
    extra = f'<div class="gg-tile-note">{note_text}</div>' if note_text else ""
    return (f'<div class="gg-tile"><div class="gg-tile-label">{label}</div>'
            f'<div class="gg-tile-value">{value}</div>{extra}</div>')


def tiles(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        col.markdown(tile(label, value, sub), unsafe_allow_html=True)


def hero(value: str, unit: str, label: str) -> None:
    st.markdown(
        f'<div class="gg-hero">{value}<span class="gg-hero-unit"> {unit}</span></div>'
        f'<div class="gg-caption">{label}</div>',
        unsafe_allow_html=True,
    )


def banner(kind: str, title: str, body: str, icon: str = "") -> None:
    icons = {"alert": "▲", "caution": "■", "ok": "●"}
    st.markdown(
        f'<div class="gg-banner gg-{kind}"><span class="gg-icon">'
        f'{icon or icons.get(kind, "●")}</span>'
        f"<span><b>{title}</b> {body}</span></div>",
        unsafe_allow_html=True,
    )


def table(df: pd.DataFrame | None, missing: str = "Not run yet.") -> bool:
    """Render a results table, or explain that its experiment has not run."""
    if df is None or df.empty:
        st.info(missing)
        return False
    st.dataframe(df, use_container_width=True, hide_index=True)
    return True


def disclaimer() -> None:
    with st.expander("What this is not"):
        st.markdown(
            """
- **Not a medical device, and not tested on anyone.** Every number here comes
  from replaying recorded traces. Nothing has been evaluated prospectively.
- **It does not recommend insulin.** The output is a glucose forecast and a
  low-glucose warning; nothing computes a dose.
- **The forecast sees a limited slice of reality.** The shipped model reads CGM
  only. Experiments that add insulin and carbohydrate records are reported
  separately and have not replaced it.
- **It refuses rather than guesses.** A gap longer than 15 minutes in the last
  two hours produces no forecast at all.
- **The alarm is a trade-off, not a setting with a right answer.** Every extra
  low it catches costs false alarms.
"""
        )
