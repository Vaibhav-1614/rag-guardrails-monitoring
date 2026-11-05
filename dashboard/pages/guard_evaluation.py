"""Guard evaluation results (eval/results/guard_eval_*.json)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
st.set_page_config(page_title="Guard evaluation", layout="wide")
st.title("Guard evaluation")

files = sorted((ROOT / "eval" / "results").glob("guard_eval_*.json"))
if not files:
    st.warning("No results yet. Run `python eval/run_guard_eval.py`.")
    st.stop()
choice = st.selectbox("Report", [f.stem for f in files], index=len(files) - 1)
rep = json.loads((ROOT / "eval" / "results" / f"{choice}.json").read_text())
m = rep["meta"]
st.caption(f"config `{m['config']}` · embedder `{m['embedder']}` · injection classifier `{m['injection_classifier'] or 'rules only'}` · "
           f"NLI `{m['nli_backend']}` · generator `{m['generator']}`")

rows = []
for name, g in rep["guards"].items():
    rows.append({"guard": name, "precision": g.get("precision"), "recall": g.get("recall"), "f1": g.get("f1"),
                 "golden FP": g.get("golden_fp")})
st.subheader("Per-guard detection")
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

s = rep["system"]
cols = st.columns(5)
cols[0].metric("Golden false blocks", s["golden_answerable_blocked"])
cols[1].metric("Injection ASR", f"{s['injection_asr_enforced']:.0%}")
cols[2].metric("Over-refusal", f"{s['over_refusal_rate']:.0%}")
cols[3].metric("Unsupported-number rate", f"{s['golden_unsupported_number_rate']:.0%}")
cols[4].metric("Guard overhead p95", f"{rep['latency']['guard_overhead_p95_ms']:.0f} ms")

if "gates" in rep:
    st.subheader("CI gates")
    st.dataframe(pd.DataFrame(rep["gates"]), hide_index=True, use_container_width=True)

st.subheader("Charts")
for img in ("pr_curves.png", "numeric_tolerance.png", "guard_latency.png"):
    p = ROOT / "docs" / "assets" / img
    if p.exists():
        st.image(str(p))
