"""Streamlit dashboard for the Conjunction Risk Filter.

Run:  streamlit run dashboard/app.py
Upload a CelesTrak SOCRATES CSV in the sidebar, or use the bundled sample.
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from crf.socrates import load_socrates, triage  # noqa: E402

st.set_page_config(page_title="Conjunction Risk Filter", layout="wide")
st.title("Conjunction Risk Filter")
st.caption("Ranks satellite collision alerts as 'act', 'check data first' or 'probably noise'. "
           "Ranking is by miss distance; the flags are triage hints, not validated predictors.")

with st.sidebar:
    up = st.file_uploader("SOCRATES CSV", type="csv")
    max_miss = st.slider("Treat alerts closer than (km) as close", 0.05, 5.0, 0.5, 0.05)
    stale = st.slider("Orbit file older than (days) = stale  [assumption]", 1.0, 10.0, 3.0, 0.5)
    active_only = st.checkbox("Only pairs with an active object", True)
    hide_family = st.checkbox("Hide same-constellation pairs", False)

df = load_socrates(up if up else ROOT / "sample" / "socrates_sample.csv")
t = triage(df, max_miss_km=max_miss, stale_days=stale, active_only=active_only, hide_same_family=hide_family)

c = t["label"].value_counts()
a, b, n, tot = st.columns(4)
a.metric("Act", int(c.get("act", 0)))
b.metric("Check data first", int(c.get("check data first", 0)))
n.metric("Probably noise", int(c.get("probably noise", 0)))
tot.metric("Total alerts", len(t))

show = st.multiselect("Show", ["act", "check data first", "probably noise"], ["act", "check data first"])
cols = ["rank", "label", "name1", "name2", "tca", "miss_km", "speed_kms", "max_dse", "max_prob", "dilution", "same_family"]
st.dataframe(t[t["label"].isin(show)][cols], use_container_width=True, hide_index=True)
st.caption("MAX_PROB is SOCRATES' own worst-case bound, shown for comparison. DILUTION is as published by CelesTrak.")
