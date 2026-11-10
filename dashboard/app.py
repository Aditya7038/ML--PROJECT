"""Streamlit dashboard for the Conjunction Risk Filter.

Run:  streamlit run dashboard/app.py

Where the data comes from, in order:
  1. FastAPI backend (http://127.0.0.1:8000, override with CRF_API_URL), if it is running.
  2. Otherwise this page runs the same backend code itself and downloads the latest CelesTrak
     SOCRATES file (re-downloaded every CRF_REFRESH_HOURS, default 6). No server needed.
  3. If the download fails, the last good download, then the bundled sample, with the reason shown.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from crf.service import DEFAULT_URL, Service  # noqa: E402

API = os.environ.get("CRF_API_URL", "http://127.0.0.1:8000")
CYAN, PINK, AMBER, TEAL, PURPLE, MUTED, CREAM = "#00D1FF", "#FF3B6B", "#FFB020", "#17E3B4", "#9B7BFF", "#8EA0BF", "#EAF0FA"
LABEL_COLOR = {"act": PINK, "check data first": AMBER, "probably noise": TEAL}

st.set_page_config(page_title="Conjunction Risk Filter", page_icon="🛰️", layout="wide")
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&display=swap');
.stApp {{ background: radial-gradient(1200px 600px at 80% -10%, #0d2a4d55, transparent),
          radial-gradient(900px 500px at -10% 40%, #1a0f3a55, transparent), #050b1a; }}
section[data-testid="stSidebar"] {{ background: #07101f; border-right: 1px solid #14264a; }}
.block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
h1,h2,h3 {{ letter-spacing: -0.01em; }}
.orb {{ font-family: 'Orbitron', sans-serif; }}
.hero {{ background: linear-gradient(135deg, #0b1d3a 0%, #0a1226 60%); border: 1px solid #1b3a66; border-radius: 18px;
         padding: 28px 34px; box-shadow: 0 0 40px #00d1ff14, inset 0 0 60px #00d1ff08; margin-bottom: 18px; }}
.hero h1 {{ font-family: 'Orbitron', sans-serif; font-weight: 800; font-size: 2.5rem; margin: 0 0 4px; color: {CREAM}; }}
.hero h1 span {{ color: {CYAN}; text-shadow: 0 0 18px #00d1ff88; }}
.hero p {{ color: {MUTED}; margin: 0 0 14px; font-size: 1.02rem; }}
.badge {{ display:inline-flex; align-items:center; gap:8px; padding: 7px 14px; border-radius: 10px; margin: 0 8px 6px 0;
          background: #0a1a30; border: 1px solid #1b3a66; color: {CREAM}; font-size: .85rem; }}
.dot {{ width:9px; height:9px; border-radius:50%; display:inline-block; }}
.kpi {{ background: #0a1428; border: 1px solid #1b3a66; border-radius: 14px; padding: 16px 20px; height: 100%; }}
.kpi .k {{ color: {MUTED}; font-size: .74rem; text-transform: uppercase; letter-spacing: .12em; }}
.kpi .v {{ font-family: 'Orbitron', sans-serif; font-size: 2.2rem; font-weight: 800; line-height: 1.25; }}
.kpi .n {{ color: {MUTED}; font-size: .82rem; }}
.mcard {{ background: #0a1428; border: 1px solid #1b3a66; border-radius: 14px; padding: 16px 18px; height: 100%; min-height: 250px; }}
.mcard .tag {{ font-family: 'Orbitron', sans-serif; font-size: .78rem; letter-spacing: .1em; }}
.mcard .name {{ font-family: 'Orbitron', sans-serif; font-size: 1.12rem; font-weight: 600; margin: 4px 0 6px; color: {CREAM}; }}
.mcard .what {{ color: {MUTED}; font-size: .86rem; min-height: 56px; }}
.mcard .m {{ display:flex; gap:18px; margin-top: 10px; }}
.mcard .m div {{ flex:1; }}
.mcard .m b {{ font-family: 'Orbitron', sans-serif; font-size: 1.4rem; display:block; }}
.mcard .m small {{ color: {MUTED}; font-size: .72rem; }}
.chip {{ display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .72rem; margin-top: 8px; }}
.callout {{ background: #0a1428; border-left: 4px solid {AMBER}; border-radius: 8px; padding: 14px 18px; margin: 8px 0 18px; }}
.stTabs [data-baseweb="tab"] {{ font-size: .98rem; }}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_service() -> Service:
    return Service(url=os.environ.get("CRF_SOCRATES_URL", DEFAULT_URL),
                   cache=ROOT / "cache" / "socrates_latest.csv",
                   sample=ROOT / "sample" / "socrates_sample.csv",
                   results=ROOT / "results",
                   refresh_hours=float(os.environ.get("CRF_REFRESH_HOURS", "6")),
                   timeout=20.0)


def api_get(path, **params):
    try:
        r = requests.get(API + path, params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def kpi(col, key, value, note="", color=CREAM):
    col.markdown(f'<div class="kpi"><div class="k">{key}</div><div class="v" style="color:{color}">{value}</div>'
                 f'<div class="n">{note}</div></div>', unsafe_allow_html=True)


def badge(text, color=None):
    d = f'<span class="dot" style="background:{color}"></span>' if color else ""
    return f'<span class="badge">{d}{text}</span>'


def style_fig(fig, height=380):
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(8,16,34,0.7)", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.12, x=0))
    return fig


svc = get_service()

# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown('<div class="orb" style="font-size:1.1rem;margin-bottom:8px">Controls</div>', unsafe_allow_html=True)
    max_miss = st.slider("Treat pairs closer than (km) as close", 0.05, 5.0, 0.5, 0.05)
    stale = st.slider("Orbit file older than (days) = stale", 1.0, 10.0, 3.0, 0.5,
                      help="Assumption, not a validated value. Old orbit files make the miss distance unreliable.")
    active_only = st.checkbox("Only pairs with an active object", True)
    hide_family = st.checkbox("Hide same-constellation pairs", False)
    refresh_clicked = st.button("Refresh data now", use_container_width=True)

# ---------------- data: API first, else run the backend code here ----------------
health = api_get("/health")
mode = "api" if health else "local"
if refresh_clicked:
    if mode == "api":
        try:
            requests.post(API + "/refresh", timeout=130)
        except requests.RequestException:
            pass
    else:
        svc.refresh()
elif mode == "local":
    svc.ensure_fresh()
if mode == "api":
    health = api_get("/health") or health
else:
    health = svc.health()

# ---------------- hero ----------------
live = health["source"] in ("celestrak",) and health["age_hours"] is not None
parts = []
if mode == "api":
    parts.append(badge("FastAPI backend connected", TEAL))
else:
    parts.append(badge("Backend: running inside this page", CYAN))
if live:
    parts.append(badge(f"Live CelesTrak SOCRATES · fetched {health['age_hours']:.1f} h ago", TEAL))
else:
    parts.append(badge("Showing bundled sample, not live data", AMBER))
nxt = health.get("next_refresh_in_hours")
parts.append(badge(f"refreshes every {health['refresh_every_hours']:g} h"
                   + (f" · next in {nxt:.1f} h" if nxt is not None else "")))
parts.append(badge(f"{health['rows']:,} conjunctions loaded"))
if health.get("last_error"):
    parts.append(badge("Last download failed: " + health["last_error"][:70], PINK))
st.markdown(f"""<div class="hero"><h1>CONJUNCTION <span>RISK FILTER</span></h1>
<p>Turns thousands of satellite collision alerts into a short list: act now, get fresh data first, or probably noise.</p>
{''.join(parts)}</div>""", unsafe_allow_html=True)

tab_alerts, tab_models, tab_check, tab_how = st.tabs(
    ["Live alerts", "Models and results", "Why 'check data first'", "How it works"])

# ---------------- alerts ----------------
with tab_alerts:
    q = dict(max_miss_km=max_miss, stale_days=stale, active_only=active_only, hide_same_family=hide_family, limit=1000)
    res = (api_get("/alerts", **q) if mode == "api" else None) or svc.alerts(**q)
    counts, total, rows = res["counts"], res["total"], pd.DataFrame(res["alerts"])

    act, chk, noise = counts.get("act", 0), counts.get("check data first", 0), counts.get("probably noise", 0)
    review = act + chk
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Act", f"{act:,}", "close, orbit data fresh", PINK)
    kpi(c2, "Check data first", f"{chk:,}", "close, but orbit data is old", AMBER)
    kpi(c3, "Probably noise", f"{noise:,}", "not close", TEAL)
    kpi(c4, "Alerts to review", f"{review:,}", f"of {total:,} total ({1 - review / max(total, 1):.1%} fewer)", CYAN)
    st.caption("Counts depend on the two sliders, which are assumptions. Only the distance ranking is validated "
               "(see Models and results). The bundled sample holds only the 3,000 closest pairs, so 'probably noise' "
               "is cut off there; live data has everything.")

    if not rows.empty:
        g1, g2 = st.columns(2)
        fig = go.Figure()
        for lab, color in LABEL_COLOR.items():
            s = rows[rows["label"] == lab]
            fig.add_trace(go.Scatter(
                x=s["miss_km"], y=s["speed_kms"], mode="markers", name=lab,
                marker=dict(color=color, size=7, opacity=0.8), text=s["name1"] + "  vs  " + s["name2"],
                hovertemplate="%{text}<br>miss %{x:.3f} km<br>speed %{y:.1f} km/s<extra></extra>"))
        fig.update_xaxes(type="log", title="Miss distance (km, log scale)")
        fig.update_yaxes(title="Relative speed (km/s)")
        fig.update_layout(title="Closest pairs: miss distance vs speed")
        g1.plotly_chart(style_fig(fig), use_container_width=True)

        fig2 = go.Figure()
        for lab, color in LABEL_COLOR.items():
            s = rows[rows["label"] == lab]
            fig2.add_trace(go.Histogram(x=s["max_dse"], name=lab, marker_color=color, xbins=dict(size=0.5)))
        fig2.update_layout(barmode="stack", title="How old is the orbit data? (top 1,000 alerts)")
        fig2.add_vline(x=stale, line_dash="dash", line_color=CREAM, annotation_text="stale cut-off")
        fig2.update_xaxes(title="Age of the older orbit file (days)")
        fig2.update_yaxes(title="Alerts")
        g2.plotly_chart(style_fig(fig2), use_container_width=True)

    show = st.multiselect("Show", list(LABEL_COLOR), ["act", "check data first"])
    cols = ["rank", "label", "name1", "name2", "tca", "miss_km", "speed_kms", "max_dse", "max_prob", "dilution",
            "same_family"]
    if rows.empty:
        rows = pd.DataFrame(columns=cols)
    view = rows[rows["label"].isin(show)][cols].rename(columns={
        "name1": "object 1", "name2": "object 2", "tca": "closest approach (UTC)", "miss_km": "miss (km)",
        "speed_kms": "speed (km/s)", "max_dse": "oldest orbit (days)", "max_prob": "SOCRATES max prob",
        "dilution": "dilution", "same_family": "same constellation"})
    styled = view.style.map(lambda v: f"color:{LABEL_COLOR.get(v, CREAM)}; font-weight:600", subset=["label"])
    st.dataframe(styled, hide_index=True, use_container_width=True, height=480)
    st.caption("SOCRATES max prob is CelesTrak's own worst-case bound, shown only for comparison.")

# ---------------- models ----------------
with tab_models:
    data = (api_get("/models") if mode == "api" else None) or svc.models()
    it, cv = data.get("independent_test"), data.get("cross_validation")

    st.markdown('<div class="callout"><b>Bottom line.</b> On the free fields SOCRATES has (miss distance, speed), '
                'machine learning did <b>not</b> beat the plain distance rule, so the live alerts rank by distance. '
                'Adding covariance and tracking-quality data, which operators get in their own CDMs, '
                'clearly improves results. That is the case for a CDM-based product.</div>', unsafe_allow_html=True)

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

        main = [r for r in recs if "alerts only" not in r["model"]]
        for i in range(0, len(main), 3):
            cols3 = st.columns(3)
            for j, r in enumerate(main[i:i + 3]):
                n = i + j + 1
                live_model = r["model"] == "Distance-only baseline"
                ref = r["model"].startswith("REFERENCE")
                color = AMBER if live_model else (PURPLE if ref else CYAN)
                chip = (f'<span class="chip" style="background:#3a2a05;color:{AMBER}">RUNNING LIVE</span>' if live_model else
                        f'<span class="chip" style="background:#2a1a55;color:{PURPLE}">NEEDS CDM DATA</span>' if ref else
                        '<span class="chip" style="background:#10243f;color:#8EA0BF">tested, not used</span>')
                pr, rm = r["pr_auc"], r["removed"]
                cols3[j].markdown(
                    f'<div class="mcard"><div class="tag" style="color:{color}">MODEL {n}</div>'
                    f'<div class="name">{r["model"].replace("REFERENCE: ", "")}</div>'
                    f'<div class="what">{r["what"]}<br><i>Features: {r["features"]}</i></div>'
                    f'<div class="m"><div><b style="color:{color}">{pr[0]:.2f}</b><small>PR-AUC<br>[{pr[1]:.2f}-{pr[2]:.2f}]</small></div>'
                    f'<div><b style="color:{color}">{rm[0]:.0%}</b><small>false alarms removed<br>[{rm[1]:.0%}-{rm[2]:.0%}]</small></div></div>'
                    f'{chip}</div>', unsafe_allow_html=True)
            st.write("")

        df = pd.DataFrame([{"Model": r["model"].replace("REFERENCE: ", "REF: "), "removed": r["removed"][0],
                            "lo": r["removed"][1], "hi": r["removed"][2], "pr": r["pr_auc"][0]} for r in recs])
        df = df.sort_values("removed")
        colors = [AMBER if m == "Distance-only baseline" else PURPLE if m.startswith("REF") else CYAN for m in df["Model"]]
        fig = go.Figure(go.Bar(
            x=df["removed"], y=df["Model"], orientation="h", marker_color=colors,
            error_x=dict(type="data", symmetric=False, array=(df["hi"] - df["removed"]).tolist(),
                         arrayminus=(df["removed"] - df["lo"]).tolist(), color=CREAM),
            hovertemplate="%{y}<br>removed %{x:.1%}<extra></extra>"))
        fig.update_xaxes(tickformat=".0%", range=[0, 1], title="False alarms removed at 90% recall (higher is better)")
        fig.update_layout(title="All models, same test")
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)
        st.caption("Amber = what the live alerts use. Purple = needs CDM data. Whiskers = 95% bootstrap interval. "
                   "'False alarms removed' means: keep 90% of the truly dangerous events, discard this share of the harmless ones.")

    with st.expander("What these numbers mean, and honest limits"):
        st.markdown("""
- **PR-AUC**: how well a model ranks the truly dangerous events above harmless ones, when dangerous ones are rare. Higher is better.
- **Label**: an event is "high-risk" if its final risk was at least 1 in 1,000,000 (threshold to be confirmed on the Kelvins page).
- **time_to_tca is excluded**: in ESA train data, earlier alerts look riskier, but the test data has almost only 2-3 day alerts and the pattern vanishes.
- We looked at the test results before dropping that column, so the test is no longer fully untouched.
- The stale-data flag and both sliders are heuristics, not validated.
""")

# ---------------- why check data first ----------------
with tab_check:
    st.markdown("### Why do some alerts say 'check data first'?")
    st.markdown(f"""
It is because of **old orbit data**. Each SOCRATES row says how old the orbit file of each object is
(the `DSE` columns: days since the orbit was measured). A satellite or debris piece keeps moving, and small errors
in an old orbit grow every day. So the further the file is from today, the less you can trust the predicted miss distance.

- **act**: the pair is close **and** both orbit files are newer than the stale cut-off (default {stale:g} days).
- **check data first**: the pair is close, **but** the older file is at least {stale:g} days old. The real miss could be larger or smaller. Get a fresh orbit before spending fuel.
- **probably noise**: not close.

The cut-off is a slider because it is an assumption. We did not validate it on real outcomes, only the distance ranking.
""")
    if not rows.empty and (rows["label"] == "check data first").any():
        ex = rows[rows["label"] == "check data first"].iloc[0]
        st.markdown(f'<div class="callout">Example from the table now: <b>{ex["name1"]}</b> vs <b>{ex["name2"]}</b> '
                    f'is only {ex["miss_km"]:.3f} km away, but its older orbit file is {ex["max_dse"]:.1f} days old, '
                    f'so it is flagged instead of marked "act".</div>', unsafe_allow_html=True)

# ---------------- how it works ----------------
with tab_how:
    st.markdown("### How the pieces connect")
    st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#0a1428", color="#1b3a66",
  fontcolor="#EAF0FA", fontname="Helvetica"]; edge [color="#8EA0BF", fontcolor="#8EA0BF", fontname="Helvetica"];
  celestrak [label="CelesTrak SOCRATES\\n(new file about 3 times a day)"];
  api [label="Backend (FastAPI)\\nfetch, cache, rank, flag", color="#00D1FF"];
  dash [label="This dashboard"];
  other [label="Other tools / operators\\n(through the API)"];
  esa [label="ESA dataset\\n(real CDMs with outcomes)"];
  val [label="Validation scripts\\nevaluate.py, train_final.py"];
  res [label="results/*.json"];
  celestrak -> api [label="every 6 h"];
  api -> dash [label="/alerts  /health"];
  api -> other [label="/alerts"];
  esa -> val -> res; res -> api [label="/models"];
}""")
    st.markdown("""
**Why a backend, when the dashboard could just read a file?**
- SOCRATES publishes a new file about three times a day. The backend downloads it on a schedule, so nobody uploads anything by hand.
- If a download fails (network, CelesTrak down, bad file), it keeps the last good copy and shows the error instead of breaking.
- One place does the ranking. The dashboard, a script, or an operator's own system all call the same `/alerts` endpoint.
- It serves the validation results too (`/models`), so the accuracy shown here comes from the real test runs, not typed-in numbers.

**Two ways to run it:** start `uvicorn crf.api:app --app-dir src` and the dashboard uses it (badge says *FastAPI backend connected*).
If the API is not running, this page runs the same backend code itself, so it still downloads live data.
Endpoints (Swagger docs at `/docs`): `GET /alerts`, `GET /health`, `GET /models`, `POST /refresh`.
""")
