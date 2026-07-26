"""GlucoGuard — entry point.

Run:  streamlit run app.py

Pages are declared here rather than inferred from filenames, so the sidebar
reads as a route through the argument: what it does, then the evidence, then
how to reproduce it.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="GlucoGuard", page_icon="\U0001FA78", layout="wide")

PAGES = [
    ("pages/1_Overview.py", "Overview", ":material/home:"),
    ("pages/2_Patient_explorer.py", "Patient explorer", ":material/monitoring:"),
    ("pages/3_Models.py", "Models", ":material/network_node:"),
    ("pages/4_Alarm.py", "Alarm", ":material/notifications_active:"),
    ("pages/5_Calibration.py", "Calibration", ":material/tune:"),
    ("pages/6_Generalisation.py", "Generalisation", ":material/public:"),
    ("pages/7_Inputs.py", "Inputs", ":material/input:"),
    ("pages/8_Live.py", "Live", ":material/sensors:"),
    ("pages/9_Method.py", "Method", ":material/science:"),
]

nav = st.navigation([
    st.Page(path, title=title, icon=icon, default=(i == 0))
    for i, (path, title, icon) in enumerate(PAGES)
])
nav.run()
