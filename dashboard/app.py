"""Streamlit dashboard for the Conjunction Risk Filter.

Run:  streamlit run dashboard/app.py
It reads from the FastAPI backend (http://127.0.0.1:8000, override with CRF_API_URL).
If the backend is not running, it falls back to the bundled sample so the demo still works.
"""
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from crf.service import read_results  # noqa: E402
from crf.socrates import load_socrates, triage  # noqa: E402

API = os.environ.get("CRF_API_URL", "http://127.0.0.1:8000")
AMBER, NAVY, PANEL, CREAM, GREEN, RED = "#E8A33D", "#0E1624", "#162236", "#F4F1EA", "#4CC38A", "#E5604D"
LABEL_COLOR = {"act": RED, "check data first": AMBER, "probably noise": "#6B7A90"}

st.set_page_config(page_title="Conjunction Risk Filter", layout="wide")
st.markdown(f"""
<style>
.block-container {{ padding-top: 2rem; max-width: 1250px; }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}
.hero {{ font-size: 2.4rem; font-weight: 700; margin-bottom: .1rem; }}
.hero span {{ color: {AMBER}; }}
.sub {{ color: #9AA7BA; margin-bottom: 1.2rem; }}
.pill {{ display:inline-block; padding: 4px 12px; border-radius: 999px; font-size: .82rem; margin-right: 8px;
        background: {PANEL}; border: 1px solid #243450; color: {CREAM}; }}
.dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }}
.card {{ background: {PANEL}; border: 1px solid #243450; border-radius: 14px; padding: 16px 20px; }}
.card .k {{ color: #9AA7BA; font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; }}
.card .v {{ font-size: 2.1rem; font-weight: 700; line-height: 1.2; }}
.card .n {{ color: #9AA7BA; font-size: .8rem; }}
.callout {{ background: {PANEL}; border-left: 4px solid {AMBER}; border-radius: 8px; padding: 14px 18px; margin: 8px 0 18px; }}
</style>""", unsafe_allow_html=True)


def api_get(path, **params):
    try:
        r = requests.get(API + path, params=params, timeout=8)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def card(col, key, value, note="", color=CREAM):
    col.markdown(f'<div class="card"><div class="k">{key}</div><div class="v" style="color:{color}">{value}</div>'
                 f'<div class="n">{note}</div></div>', unsafe_allow_html=True)


# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("### Controls")
    up = st.file_uploader("Use my own SOCRATES CSV (optional)", type="csv")
    max_miss = st.slider("Treat pairs closer than (km) as close", 0.05, 5.0, 0.5, 0.05)
    stale = st.slider("Orbit file older than (days) = stale", 1.0, 10.0, 3.0, 0.5,
                      help="Assumption, not a validated value. Old orbit files make miss distance unreliable.")
    active_only = st.checkbox("Only pairs with an active object", True)
    hide_family = st.checkbox("Hide same-constellation pairs", False)

health = None if up else api_get("/health")
mode = "upload" if up else ("api" if health else "local")

# ---------------- header ----------------
st.markdown('<div class="hero">Conjunction <span>Risk Filter</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Turns thousands of satellite collision alerts into a short list: '
            'act now, get fresh data first, or probably noise.</div>', unsafe_allow_html=True)

if mode == "api":
    age = health["age_hours"]
    when = f"data fetched {age:.1f} h ago" if age is not None else f"data: {health['source']}"
    nxt = health["next_refresh_in_hours"]
    st.markdown(f'<span class="pill"><span class="dot" style="background:{GREEN}"></span>Backend connected</span>'
                f'<span class="pill">{when}</span>'
                f'<span class="pill">source: {health["source"]}</span>'
                + (f'<span class="pill">next refresh in {nxt:.1f} h</span>' if nxt is not None else "")
                + (f'<span class="pill" style="border-color:{RED}">last download error: {health["last_error"][:60]}</span>'
                   if health["last_error"] else ""), unsafe_allow_html=True)
else:
    why = "using your uploaded file" if mode == "upload" else "backend offline, using bundled sample"
    st.markdown(f'<span class="pill"><span class="dot" style="background:{AMBER}"></span>Local mode</span>'
                f'<span class="pill">{why}</span>', unsafe_allow_html=True)

tab_alerts, tab_models, tab_how = st.tabs(["Alerts", "Models and accuracy", "How it works"])

# ---------------- alerts ----------------
with tab_alerts:
    if mode == "api":
        res = api_get("/alerts", max_miss_km=max_miss, stale_days=stale, active_only=active_only,
                      hide_same_family=hide_family, limit=1000) or {"counts": {}, "total": 0, "alerts": []}
        counts, total, rows = res["counts"], res["total"], pd.DataFrame(res["alerts"])
    else:
        t = triage(load_socrates(up if up else ROOT / "sample" / "socrates_sample.csv"), max_miss_km=max_miss,
                   stale_days=stale, active_only=active_only, hide_same_family=hide_family)
        counts, total = t["label"].value_counts().to_dict(), len(t)
        rows = t.head(1000).assign(tca=lambda x: x["tca"].dt.strftime("%Y-%m-%d %H:%M:%S"))

    act, chk, noise = counts.get("act", 0), counts.get("check data first", 0), counts.get("probably noise", 0)
    review = act + chk
    c1, c2, c3, c4 = st.columns(4)
    card(c1, "Act", f"{act:,}", "close, orbit data fresh", RED)
    card(c2, "Check data first", f"{chk:,}", "close, orbit data stale", AMBER)
    card(c3, "Probably noise", f"{noise:,}", "not close", "#9AA7BA")
    card(c4, "Alerts to review", f"{review:,}", f"of {total:,} total ({(1 - review / max(total, 1)):.1%} fewer)", GREEN)
    st.caption("Counts depend on the two sliders, which are assumptions. Only the distance ranking is validated "
               "(see the Models tab). If you use the bundled sample, 'probably noise' is cut off: the sample "
               "holds only the 3,000 closest pairs.")

    if rows.empty:
        rows = pd.DataFrame(columns=["rank", "label", "name1", "name2", "tca", "miss_km", "speed_kms", "max_dse",
                                     "max_prob", "dilution", "same_family"])
    show = st.multiselect("Show", list(LABEL_COLOR), ["act", "check data first"])
    cols = ["rank", "label", "name1", "name2", "tca", "miss_km", "speed_kms", "max_dse", "max_prob", "dilution",
            "same_family"]
    view = rows[rows["label"].isin(show)][cols].rename(columns={
        "name1": "object 1", "name2": "object 2", "tca": "closest approach (UTC)", "miss_km": "miss (km)",
        "speed_kms": "speed (km/s)", "max_dse": "oldest orbit (days)", "max_prob": "SOCRATES max prob",
        "dilution": "dilution", "same_family": "same constellation"})
    styled = view.style.map(lambda v: f"color:{LABEL_COLOR.get(v, CREAM)}; font-weight:600", subset=["label"])
    st.dataframe(styled, hide_index=True, use_container_width=True, height=520)
    st.caption("SOCRATES max prob is CelesTrak's own worst-case bound, shown only for comparison.")

# ---------------- models ----------------
with tab_models:
    data = api_get("/models") if mode == "api" else None
    data = data or read_results(ROOT / "results")
    it, cv = data.get("independent_test"), data.get("cross_validation")

    st.markdown(f'<div class="callout"><b>Bottom line.</b> On the free fields SOCRATES has (miss distance, speed), '
                f'machine learning did <b>not</b> beat the plain distance rule, so the live demo ranks by distance. '
                f'Adding covariance and tracking-quality data, which operators get in their own CDMs, '
                f'clearly improves results. That is the case for a CDM-based product.</div>', unsafe_allow_html=True)

    if not it:
        st.info("No results found. Run src/crf/train_final.py to create results/independent_test.json.")
    else:
        choice = st.radio("Evaluation", ["Independent ESA test", "Cross-validation (ESA train data)"], horizontal=True)
        if choice.startswith("Independent"):
            recs, n_ev, n_hi = it["models"], it["events"], it["high_risk"]
            st.caption(f"ESA's own test files: {n_ev:,} events, {n_hi} high-risk. Models trained on all train events.")
        else:
            sub = cv["subsets"]["ALL EVENTS"]
            recs, n_ev, n_hi = sub["models"], sub["events"], sub["high_risk"]
            st.caption(f"5-fold cross-validation: {n_ev:,} events, {n_hi} high-risk. Train data has fewer high-risk "
                       f"events than the test set, so compare models within one view, not across views.")

        df = pd.DataFrame([{
            "Model": r["model"], "What it does": r["what"], "Features": r["features"],
            "PR-AUC": r["pr_auc"][0], "pr_lo": r["pr_auc"][1], "pr_hi": r["pr_auc"][2],
            "removed": r["removed"][0], "rm_lo": r["removed"][1], "rm_hi": r["removed"][2],
            "In live demo": "yes" if r["model"] == "Distance-only baseline" else ""} for r in recs])

        base = alt.Chart(df).encode(y=alt.Y("Model:N", sort=list(df.sort_values("removed", ascending=False)["Model"]), title=None, axis=alt.Axis(labelLimit=330)))
        bars = base.mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("removed:Q", title="False alarms removed at 90% recall (higher is better)",
                    axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.condition(alt.datum.Model == "Distance-only baseline", alt.value(AMBER), alt.value("#4F7CAC")),
            tooltip=["Model", alt.Tooltip("removed:Q", format=".1%"), alt.Tooltip("PR-AUC:Q", format=".3f")])
        err = base.mark_errorbar(color=CREAM).encode(x="rm_lo:Q", x2="rm_hi:Q")
        st.altair_chart((bars + err).properties(height=300), use_container_width=True)
        st.caption("Amber = what the live demo runs. Whiskers = 95% bootstrap interval. "
                   "'False alarms removed' means: keep 90% of the truly dangerous events, discard this share of the harmless ones.")

        out = df.assign(**{"False alarms removed": df.apply(
            lambda r: f"{r['removed']:.0%}  [{r['rm_lo']:.0%}-{r['rm_hi']:.0%}]", axis=1),
            "PR-AUC (95% interval)": df.apply(lambda r: f"{r['PR-AUC']:.2f}  [{r['pr_lo']:.2f}-{r['pr_hi']:.2f}]", axis=1)})
        st.dataframe(out[["Model", "What it does", "Features", "PR-AUC (95% interval)", "False alarms removed",
                          "In live demo"]], hide_index=True, use_container_width=True)

    with st.expander("What these numbers mean, and honest limits"):
        st.markdown("""
- **PR-AUC**: how well a model ranks the truly dangerous events above harmless ones, when dangerous ones are rare. Higher is better.
- **Label**: an event is "high-risk" if its final risk was at least 1 in 1,000,000 (threshold to be confirmed on the Kelvins page).
- **time_to_tca is excluded**: in ESA train data, earlier alerts look riskier, but the test data has almost only 2-3 day alerts and the pattern vanishes.
- We looked at the test results before dropping that column, so the test is no longer fully untouched.
- The stale-data flag and both sliders are heuristics, not validated.
""")

# ---------------- how it works ----------------
with tab_how:
    st.markdown("### How the pieces connect")
    st.graphviz_chart(f"""
digraph G {{
  rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="{PANEL}", color="#243450",
  fontcolor="{CREAM}", fontname="Helvetica"]; edge [color="#9AA7BA", fontcolor="#9AA7BA", fontname="Helvetica"];
  celestrak [label="CelesTrak SOCRATES\\n(new file about 3 times a day)"];
  api [label="FastAPI backend\\nfetch, cache, rank, flag", color="{AMBER}"];
  dash [label="This dashboard"];
  other [label="Other tools / operators\\n(through the API)"];
  esa [label="ESA dataset\\n(real CDMs with outcomes)"];
  val [label="Validation scripts\\nevaluate.py, train_final.py"];
  res [label="results/*.json"];
  celestrak -> api [label="every 6 h"];
  api -> dash [label="/alerts  /health"];
  api -> other [label="/alerts"];
  esa -> val -> res; res -> api [label="/models"];
}}""")
    st.markdown("""
**Why a backend, when the dashboard could just read a file?**
- SOCRATES publishes a new file about three times a day. The backend downloads it on a schedule, so the dashboard never needs a manual upload.
- If a download fails (network, CelesTrak down, bad file), it keeps the last good copy and shows the error instead of breaking.
- One place does the ranking. The dashboard, a script, or an operator's own system all call the same `/alerts` endpoint.
- It serves the validation results too (`/models`), so the accuracy shown here comes from the real test runs, not typed-in numbers.

**Endpoints** (Swagger docs at `/docs`): `GET /alerts`, `GET /health`, `GET /models`, `POST /refresh`.
""")
