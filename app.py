"""GlucoGuard — entry point.

Run:  streamlit run app.py

Pages are declared here rather than inferred from filenames, so the sidebar
reads as a route through the argument: what it does, then the evidence, then
how to reproduce it.

They live in `views/` rather than `pages/` on purpose: Streamlit auto-discovers
a `pages/` directory and would list every file in it *alongside* the navigation
declared below, so the sidebar ends up with each page twice and this script
showing up as "app".
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="GlucoGuard", page_icon="\U0001FA78", layout="wide")

PAGES = [
    ("views/1_Overview.py", "Overview", ":material/home:"),
    ("views/2_Patient_explorer.py", "Patient explorer", ":material/monitoring:"),
    ("views/3_Models.py", "Models", ":material/network_node:"),
    ("views/4_Alarm.py", "Alarm", ":material/notifications_active:"),
    ("views/5_Calibration.py", "Calibration", ":material/tune:"),
    ("views/6_Generalisation.py", "Generalisation", ":material/public:"),
    ("views/7_Inputs.py", "Inputs", ":material/input:"),
    ("views/8_Live.py", "Live", ":material/sensors:"),
    ("views/9_Method.py", "Method", ":material/science:"),
]

nav = st.navigation([
    st.Page(path, title=title, icon=icon, default=(i == 0))
    for i, (path, title, icon) in enumerate(PAGES)
])
nav.run()
