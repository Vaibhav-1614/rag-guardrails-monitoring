"""Monitoring: nightly drift reports (monitoring/reports/) or the simulated validation runs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
st.set_page_config(page_title="Monitoring", layout="wide")
st.title("Drift monitoring")

prod = sorted((ROOT / "monitoring" / "reports").glob("20*.json"))
sim = sorted((ROOT / "monitor" / "validation" / "reports").glob("*.json"))
sources = {f"production · {p.stem}": p for p in prod} | {f"simulated · {p.stem}": p for p in sim}
if not sources:
    st.warning("No reports. Run `python -m monitor.run` (production) or `python -m monitor.validate_detectors` (simulation).")
    st.stop()
if not prod:
    st.info("No production reports in this checkout (traffic is low); showing drift detectors validated against simulated scenarios.")
label = st.selectbox("Report", list(sources), index=len(prod) - 1 if prod else 0)
rep = json.loads(sources[label].read_text())

st.caption(f"as of {rep['as_of']} · env `{rep['env']}` · window {rep['window'][0]}..{rep['window'][1]} ({rep['n_window']} requests) · "
           f"reference `{rep['reference']['version']}` · embedder `{rep['embedder']}`")
det = rep["detectors"]
cols = st.columns(4)
for i, (name, d) in enumerate(det.items()):
    cols[i % 4].metric(name.replace("_", " "), "FIRED" if d.get("fired") else ("no data" if d.get("status") else "ok"))

if rep["alerts"]:
    st.subheader("Alerts")
    st.dataframe(pd.DataFrame(rep["alerts"]), hide_index=True, use_container_width=True)

em = det.get("emerging_topics", {})
if em.get("clusters"):
    st.subheader("Emerging topics" if em.get("fired") else "Candidate novel clusters (below the alert threshold)")
    for c in em["clusters"]:
        st.markdown(f"- **{', '.join(c['top_terms'][:5])}**: {c['size']} queries ({c['share']:.0%}), e.g. _{c['examples'][0]}_")

daily = pd.DataFrame(rep["daily"])
if not daily.empty:
    st.subheader("Daily metrics")
    trig = pd.json_normalize(daily["triggers"]).add_prefix("trigger_")
    rates = trig.div(daily["n"], axis=0)
    df = pd.concat([daily.drop(columns=["triggers"]), rates], axis=1)
    groups = {
        "Traffic and refusals": ["refusal_rate", "out_of_scope_rate", "low_confidence_rate"],
        "Output behaviour": ["regeneration_rate", "flagged_rate", "unsupported_rate"],
        "Answer shape": ["citations_mean", "answer_chars_mean"],
        "Guard trigger rates": [c for c in rates.columns if rates[c].max() > 0],
        "Ops": ["latency_p95_ms", "n"],
    }
    for title, cols_ in groups.items():
        cols_ = [c for c in cols_ if c in df]
        if cols_:
            fig = px.line(df, x="day", y=cols_, markers=True, title=title, height=300)
            st.plotly_chart(fig, use_container_width=True)

if rep.get("golden"):
    st.subheader("Nightly golden-set runs (upstream model drift)")
    g = pd.DataFrame([{"run": i, **x["metrics"]} for i, x in enumerate(rep["golden"])])
    fig = px.line(g, x="run", y=[c for c in ["answered_rate", "reference_number_recall", "citation_coverage_mean",
                                            "numeric_support_mean", "unsupported_answer_rate"] if c in g], markers=True, height=320)
    st.plotly_chart(fig, use_container_width=True)
with st.expander("Raw report"):
    st.json(rep, expanded=False)
