"""Nightly drift job (design doc §8).

    python -m monitor.run --env production --golden inprocess --alert github
    python -m monitor.run --env sim-topic-shift --as-of 2026-10-14 --golden skip

Reads the last ``history_days`` of traces (Langfuse public API or local JSONL), compares the current
window with the versioned reference and with each metric's trailing band, optionally runs the golden
set against production, writes a dated JSON + Markdown report and raises alerts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from monitor import drift
from monitor.features import daily_aggregates, to_row
from monitor.golden_run import append_history, load_history
from monitor.golden_run import run as run_golden
from monitor.reference import Reference
from rag.config import load_settings, load_yaml, resolve

GUARDS = ["charset", "pii", "prompt_injection", "scope", "advice", "retrieval_conf", "context_sanitizer", "citation",
          "numeric_grounding", "nli_grounding", "advice_output", "pii_output"]


def _clean(x: Any) -> Any:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_clean(v) for v in x]
    if isinstance(x, np.floating | np.integer):
        return _clean(x.item())
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _bands(daily: list[dict], today: str, metrics: dict[str, dict], cfg: dict) -> dict[str, dict]:
    min_n = int(cfg.get("min_requests_per_day", 20))
    past = [d for d in daily if d["day"] < today and d["n"] >= min_n][-int(cfg.get("band_window", 7)):]
    cur = next((d for d in daily if d["day"] == today), None)
    out = {}
    for m, mcfg in metrics.items():
        if cur is None or cur["n"] < min_n:
            out[m] = {"status": "insufficient_data"}
            continue
        hist = [d[m] for d in past]
        min_std = float(mcfg.get("min_std", 0.0))
        if m.endswith("_rate"):
            # A daily rate over n requests can't be estimated more precisely than its binomial standard error;
            # without this floor, ordinary sampling noise at low traffic looks like drift.
            min_std = max(min_std, binomial_se(hist, cur["n"]))
        b = drift.rolling_band(hist, cur[m], k=float(cfg.get("band_k", 3.0)), min_std=min_std,
                               min_abs_change=float(mcfg.get("min_abs_change", 0.0)))
        out[m] = b.to_dict()
    return out


def binomial_se(history: list[float], n: int) -> float:
    vals = [h for h in history if h is not None and not math.isnan(h)]
    if not vals or n <= 0:
        return 0.0
    p = min(max(float(np.median(vals)), 1.0 / n), 1 - 1.0 / n)
    return math.sqrt(p * (1 - p) / n)


def compute_report(history: list[dict], as_of: date, reference: Reference, embedder, cfg: dict,
                   golden_history: list[dict] | None = None, scope_threshold: float = 0.45, env: str = "") -> dict[str, Any]:
    rows = [to_row(r) for r in history]
    today = as_of.strftime("%Y-%m-%d")
    win_start = (as_of - timedelta(days=int(cfg.get("window_days", 7)) - 1)).strftime("%Y-%m-%d")
    window = [r for r in rows if win_start <= r["day"] <= today]
    daily = daily_aggregates(rows, GUARDS)
    alerts: list[dict] = []
    det: dict[str, Any] = {}

    # --- query drift + emerging topics (embeddings of the current window vs. reference)
    qcfg, ecfg = cfg["query_drift"], cfg["emerging_topics"]
    texts = [r["question"] for r in window if r["question"]]
    if len(texts) >= int(qcfg.get("min_samples", 50)):
        ref_emb = reference.embed(embedder)
        cur_emb = embedder.encode(texts).astype(np.float32)
        c2 = drift.classifier_two_sample_test(ref_emb, cur_emb)
        mmd = drift.mmd_test(ref_emb, cur_emb, n_perm=int(qcfg.get("mmd_permutations", 200)))
        fired = c2["auc"] >= float(qcfg["c2st_auc"]) and c2["p_value"] < float(qcfg["c2st_p"])
        det["query_drift"] = {"fired": fired, "c2st": c2, "mmd": mmd, "centroid_distance": round(drift.centroid_distance(ref_emb, cur_emb), 4),
                              "n": len(texts), "embedder": embedder.name}
        if fired:
            alerts.append({"detector": "query_drift", "metric": "c2st_auc", "value": c2["auc"], "threshold": f">= {qcfg['c2st_auc']}"})
        em = drift.emerging_clusters(ref_emb, cur_emb, texts, quantile=float(ecfg["novelty_quantile"]),
                                     min_size=int(ecfg["min_cluster_size"]), min_share=float(ecfg["min_share"]))
        e_fired = em["novel_share"] >= float(ecfg["min_novel_share"]) and bool(em["clusters"])
        det["emerging_topics"] = {"fired": e_fired, **em}
        if e_fired:
            alerts.append({"detector": "emerging_topics", "metric": "novel_share", "value": em["novel_share"],
                           "threshold": f">= {ecfg['min_novel_share']}"})
    else:
        det["query_drift"] = {"fired": False, "status": "insufficient_data", "n": len(texts)}
        det["emerging_topics"] = {"fired": False, "status": "insufficient_data"}

    # --- scope drift
    scfg = cfg["scope_drift"]
    cur_scope = np.array([r["scope"] for r in window if r["scope"] is not None])
    oos_window = float(np.mean([r["scope_reason"] is not None for r in window])) if window else float("nan")
    band = _bands(daily, today, {"out_of_scope_rate": {"min_abs_change": scfg.get("min_abs_change", 0.08), "min_std": 0.02}}, cfg)["out_of_scope_rate"]
    p_scope = drift.psi(reference.scope_scores, cur_scope) if len(cur_scope) >= 20 else float("nan")
    moved = abs(oos_window - reference.out_of_scope_rate) >= float(scfg.get("min_abs_change", 0.08)) if not math.isnan(oos_window) else False
    s_fired = bool(band.get("outside")) or (not math.isnan(p_scope) and p_scope > float(scfg["psi"]) and moved)
    det["scope_drift"] = {"fired": s_fired, "psi": p_scope, "out_of_scope_rate_window": oos_window,
                          "out_of_scope_rate_reference": reference.out_of_scope_rate, "today_band": band}
    if s_fired:
        alerts.append({"detector": "scope_drift", "metric": "out_of_scope_rate", "value": round(oos_window, 4),
                       "threshold": f"PSI > {scfg['psi']} and |change| >= {scfg['min_abs_change']}, or outside band"})

    # --- retrieval drift
    rcfg = cfg["retrieval_drift"]
    cur_ret = np.array([r["retrieval_top"] for r in window if r["retrieval_top"] is not None])
    if len(cur_ret) >= int(rcfg.get("min_samples", 50)):
        p_ret = drift.psi(reference.retrieval_scores, cur_ret)
        r_fired = p_ret > float(rcfg["psi"])
        det["retrieval_drift"] = {"fired": r_fired, "psi": p_ret, "n": len(cur_ret),
                                  "low_confidence_rate_window": float(np.mean([r["refusal_reason"] == "retrieval_conf" for r in window])),
                                  "mean_window": float(cur_ret.mean()), "mean_reference": float(reference.retrieval_scores.mean())}
        if r_fired:
            alerts.append({"detector": "retrieval_drift", "metric": "retrieval_top_psi", "value": round(p_ret, 4), "threshold": f"> {rcfg['psi']}"})
    else:
        det["retrieval_drift"] = {"fired": False, "status": "insufficient_data", "n": len(cur_ret)}

    # --- output drift (rolling bands on daily metrics)
    ob = _bands(daily, today, cfg["output_drift"]["metrics"], cfg)
    o_fired = [m for m, b in ob.items() if b.get("outside")]
    det["output_drift"] = {"fired": bool(o_fired), "metrics": ob, "outside": o_fired}
    for m in o_fired:
        alerts.append({"detector": "output_drift", "metric": m, "value": round(ob[m]["value"], 4),
                       "threshold": f"band [{ob[m]['lower']:.3f}, {ob[m]['upper']:.3f}]"})

    # --- guard drift (trigger-rate spikes)
    gcfg = cfg["guard_drift"]
    min_n = int(cfg.get("min_requests_per_day", 20))
    cur_day = next((d for d in daily if d["day"] == today), None)
    past = [d for d in daily if d["day"] < today and d["n"] >= min_n][-int(cfg.get("band_window", 7)):]
    spikes = {}
    if cur_day and cur_day["n"] >= min_n and past:
        n_base = sum(d["n"] for d in past)
        for g in GUARDS:
            s = drift.rate_spike(cur_day["triggers"][g], cur_day["n"], sum(d["triggers"][g] for d in past), n_base,
                                 z_min=float(gcfg["z"]), ratio_min=float(gcfg["ratio"]), min_count=int(gcfg["min_count"]))
            spikes[g] = s
            if s["spike"]:
                alerts.append({"detector": "guard_drift", "metric": f"{g}_trigger_rate", "value": s["rate_today"],
                               "threshold": f"z >= {gcfg['z']} and >= {gcfg['ratio']}x baseline {s['rate_base']}"})
    det["guard_drift"] = {"fired": any(s["spike"] for s in spikes.values()), "guards": spikes,
                          "spiking": [g for g, s in spikes.items() if s["spike"]]}

    # --- upstream model drift (nightly golden runs)
    ucfg = cfg["upstream_model_drift"]
    gh = golden_history or []
    if len(gh) > int(ucfg.get("min_history", 5)):
        last, prev = gh[-1], gh[:-1][-14:]
        res, outside = {}, []
        for m, mcfg in ucfg["metrics"].items():
            hist = [p["metrics"].get(m) for p in prev]
            min_std = float(mcfg.get("min_std", 0.0))
            if m.endswith(("_rate", "_recall")):
                min_std = max(min_std, binomial_se(hist, int(last["metrics"].get("n", 0) or 0)))
            b = drift.rolling_band(hist, last["metrics"].get(m), k=float(ucfg.get("k", 3.0)), min_std=min_std,
                                   min_abs_change=float(mcfg.get("min_abs_change", 0.0)),
                                   min_history=int(ucfg.get("min_history", 5)))
            res[m] = b.to_dict()
            if b.outside:
                outside.append(m)
        same_code = all(p.get("code_version") == last.get("code_version") and p.get("config") == last.get("config") for p in prev[-3:])
        det["upstream_model_drift"] = {"fired": bool(outside), "metrics": res, "outside": outside, "code_unchanged": same_code,
                                       "interpretation": ("provider-side change: golden scores moved with no code/config change"
                                                          if same_code else "scores moved after a code/config change: check the last deploy")
                                       if outside else None, "runs": len(gh)}
        for m in outside:
            alerts.append({"detector": "upstream_model_drift", "metric": m, "value": res[m]["value"],
                           "threshold": f"band [{res[m]['lower']:.3f}, {res[m]['upper']:.3f}]"})
    else:
        det["upstream_model_drift"] = {"fired": False, "status": "insufficient_history", "runs": len(gh)}

    # --- ops drift
    opb = _bands(daily, today, cfg["ops_drift"]["metrics"], cfg)
    op_fired = [m for m, b in opb.items() if b.get("outside")]
    det["ops_drift"] = {"fired": bool(op_fired), "metrics": opb, "outside": op_fired}
    for m in op_fired:
        alerts.append({"detector": "ops_drift", "metric": m, "value": round(opb[m]["value"], 2),
                       "threshold": f"band [{opb[m]['lower']:.2f}, {opb[m]['upper']:.2f}]"})

    return _clean({
        "as_of": today, "env": env, "window": [win_start, today], "n_window": len(window), "n_today": cur_day["n"] if cur_day else 0,
        "reference": {"version": reference.version, "n": len(reference.texts), "description": reference.description},
        "embedder": embedder.name, "detectors": det, "alerts": alerts,
        "fired": sorted(k for k, v in det.items() if v.get("fired")),
        "daily": [{k: v for k, v in d.items()} for d in daily],
        "golden": gh[-14:],
    })


def to_markdown(rep: dict) -> str:
    lines = [f"# Drift report {rep['as_of']} ({rep['env']})", "",
             f"Window {rep['window'][0]} .. {rep['window'][1]}: {rep['n_window']} requests ({rep['n_today']} on {rep['as_of']}). "
             f"Reference `{rep['reference']['version']}` ({rep['reference']['n']} queries), embedder `{rep['embedder']}`.", "",
             "| Detector | Fired | Key numbers |", "|---|---|---|"]
    d = rep["detectors"]

    def key(name: str) -> str:
        x = d[name]
        if x.get("status"):
            return x["status"]
        if name == "query_drift":
            return f"C2ST AUC {x['c2st']['auc']} (p={x['c2st']['p_value']:.2g}), MMD p={x['mmd']['p_value']:.3f}, centroid dist {x['centroid_distance']}"
        if name == "emerging_topics":
            return f"novel share {x['novel_share']}, clusters: " + "; ".join(", ".join(c["top_terms"][:4]) for c in x["clusters"][:3])
        if name == "scope_drift":
            psi = x["psi"]
            return f"PSI {psi:.3f}, out-of-scope {x['out_of_scope_rate_window']:.3f} vs ref {x['out_of_scope_rate_reference']:.3f}" if psi is not None else "–"
        if name == "retrieval_drift":
            return f"PSI {x['psi']:.3f}, mean top score {x['mean_window']:.3f} vs ref {x['mean_reference']:.3f}"
        if name in ("output_drift", "ops_drift"):
            return "outside band: " + (", ".join(x["outside"]) or "none")
        if name == "guard_drift":
            return "spiking: " + (", ".join(x["spiking"]) or "none")
        if name == "upstream_model_drift":
            return ("outside band: " + (", ".join(x["outside"]) or "none")) + (f" ({x['interpretation']})" if x.get("interpretation") else "")
        return ""

    for name in d:
        lines.append(f"| {name} | {'**yes**' if d[name].get('fired') else 'no'} | {key(name)} |")
    if rep["alerts"]:
        lines += ["", "## Alerts", ""] + [f"- **{a['detector']}** `{a['metric']}` = {a['value']} ({a['threshold']})" for a in rep["alerts"]]
    em = d.get("emerging_topics", {})
    if em.get("clusters"):
        lines += ["", "## Emerging topics", ""]
        for c in em["clusters"]:
            lines.append(f"- {c['size']} queries ({c['share']:.0%}): {', '.join(c['top_terms'])}. e.g. \"{c['examples'][0]}\"")
    return "\n".join(lines) + "\n"


def write_report(rep: dict, out_dir: str | Path) -> Path:
    out = resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{rep['as_of']}.json").write_text(json.dumps(rep, indent=1))
    (out / f"{rep['as_of']}.md").write_text(to_markdown(rep))
    (out / "latest.json").write_text(json.dumps(rep, indent=1))
    return out / f"{rep['as_of']}.md"


def main() -> int:
    from monitor.alerts import open_github_issues, post_webhook
    from rag.embedder import get_embedder
    from rag.tracing import make_backend

    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="production")
    ap.add_argument("--as-of", default=None, help="last full day (default: yesterday UTC)")
    ap.add_argument("--source", default="auto", choices=["auto", "local", "langfuse"])
    ap.add_argument("--config", default="configs/monitoring.yaml")
    ap.add_argument("--golden", default=os.environ.get("MONITOR_TARGET_URL", "skip"),
                    help="'skip', 'inprocess', or the production base URL")
    ap.add_argument("--golden-history", default="monitoring/golden_history.jsonl")
    ap.add_argument("--out", default="monitoring/reports")
    ap.add_argument("--alert", choices=["none", "github"], default="none")
    ap.add_argument("--report-url", default=None)
    args = ap.parse_args()

    settings = load_settings()
    cfg = load_yaml(args.config)
    as_of = date.fromisoformat(args.as_of) if args.as_of else (datetime.now(UTC) - timedelta(days=1)).date()
    start = datetime.combine(as_of - timedelta(days=int(cfg["history_days"]) - 1), datetime.min.time(), UTC)
    end = datetime.combine(as_of + timedelta(days=1), datetime.min.time(), UTC)
    backend = make_backend(settings, None if args.source == "auto" else args.source)
    history = backend.read(start, end, args.env)
    if args.golden != "skip":
        append_history(run_golden(args.golden), args.golden_history)
    guards_cfg = load_yaml(settings.get("guards_config", "configs/guards/v2.yaml"))
    scope_thr = float(guards_cfg["guards"].get("scope", {}).get("threshold", 0.45))
    rep = compute_report(history, as_of, Reference.load(cfg["reference"]), get_embedder(settings), cfg,
                         load_history(args.golden_history), scope_thr, args.env)
    path = write_report(rep, args.out)
    print(to_markdown(rep))
    print(f"report -> {path}")
    if rep["alerts"] and args.alert == "github":
        urls = open_github_issues(rep["alerts"], rep["as_of"], args.report_url)
        post_webhook(rep["alerts"], rep["as_of"])
        print("alerts ->", urls)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
