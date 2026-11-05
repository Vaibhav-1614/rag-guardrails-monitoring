"""Validate the drift detectors against labelled traffic scenarios (design doc §8.3, week-4 milestone:
"each simulated scenario triggers the correct detector").

    python -m monitor.validate_detectors            # ~6 minutes on a laptop CPU

1. Build the reference window: golden questions + 7 days of simulated baseline traffic.
2. For each scenario, simulate 14 days (7 baseline + 7 scenario) and run the nightly job as of the last day.
3. Run the nightly job on each of the last 7 baseline days to count false alarms.
4. Write the detector x scenario matrix (docs/drift_validation.md), sample reports for the dashboard
   (monitor/validation/reports/) and a chart; exit 1 if an expected detector stayed quiet or baseline alerted.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from monitor.golden_run import load_history  # noqa: E402
from monitor.reference import Reference, build  # noqa: E402
from monitor.run import compute_report  # noqa: E402
from monitor.simulate_traffic import simulate  # noqa: E402
from rag.config import load_yaml, resolve  # noqa: E402

SCENARIOS = ["baseline", "topic_shift", "attack_wave", "hallucination_pressure", "model_drift"]
DETECTORS = ["query_drift", "emerging_topics", "scope_drift", "retrieval_drift", "output_drift", "guard_drift",
             "upstream_model_drift", "ops_drift"]
EXPECTED = {
    "baseline": set(),
    "topic_shift": {"scope_drift", "query_drift"},
    "attack_wave": {"guard_drift"},
    "hallucination_pressure": {"output_drift"},
    "model_drift": {"upstream_model_drift", "output_drift"},
}
# Day each scenario's change actually begins; alerts on earlier nights are false alarms.
ONSET = {"baseline": 99, "topic_shift": 7, "attack_wave": 13, "hallucination_pressure": 7, "model_drift": 9}
REF_START, SIM_START, DAYS = date(2026, 9, 1), date(2026, 10, 1), 14


def main() -> int:
    from rag.service import RAGService
    from rag.tracing import LocalBackend, Tracer

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-day", type=int, default=60)
    ap.add_argument("--golden-limit", type=int, default=30)
    ap.add_argument("--work", default="monitoring/sim")
    ap.add_argument("--rebuild-reference", action="store_true", help="overwrite monitor/reference/v1.npz")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    args = ap.parse_args()

    cfg = load_yaml("configs/monitoring.yaml")
    work = resolve(args.work)
    if work.exists():
        shutil.rmtree(work)
    backend = LocalBackend(work / "traces")
    svc = RAGService.build(trace_backend="none")
    svc.c.tracer = Tracer(backend, mask=svc.c.pipeline.mask, environment="sim")
    scope_thr = float(svc.c.pipeline.cfg("scope").get("threshold", 0.45))
    t0 = time.perf_counter()

    ref_path = resolve(cfg["reference"])
    if args.rebuild_reference or not ref_path.exists():
        simulate("baseline", REF_START, 7, args.per_day, seed=100, service=svc, env="sim-reference")
        from datetime import UTC, datetime

        recs = backend.read(datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 9, 8, tzinfo=UTC), "sim-reference")
        golden = [json.loads(line)["question"] for line in resolve("eval/golden.jsonl").read_text().splitlines() if line.strip()]
        ref = build(recs, golden, "v1", f"golden set ({len(golden)}) + 7 days of simulated baseline traffic "
                    f"({len(recs)} requests, seed 100, {REF_START}..{REF_START + timedelta(days=6)})", svc.c.embedder)
        ref.save(ref_path)
        print(f"reference v1: {len(ref.texts)} queries -> {ref_path.relative_to(ROOT)}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    reference = Reference.load(ref_path)

    from datetime import UTC, datetime

    results, series = {}, {}
    out_reports = ROOT / "monitor" / "validation" / "reports"
    out_reports.mkdir(parents=True, exist_ok=True)
    for i, scenario in enumerate(args.scenarios.split(",")):
        t = time.perf_counter()
        gh_path = work / scenario / "golden_history.jsonl"
        info = simulate(scenario, SIM_START, DAYS, args.per_day, seed=i + 1, service=svc,
                        golden_limit=args.golden_limit, golden_history=str(gh_path))
        env = info["env"]
        hist = backend.read(datetime(2026, 9, 1, tzinfo=UTC), datetime(2026, 10, 16, tzinfo=UTC), env)
        golden = load_history(gh_path)
        # Emulate the nightly cron: run the job as of every night of the scenario period (days 7..13).
        nightly, first_night, alerts, rep = [], {}, [], None
        for d in range(DAYS // 2, DAYS):
            day = SIM_START + timedelta(days=d)
            g = [x for x in golden if x.get("simulated_day", 0) <= d]
            rep = compute_report([h for h in hist if h["timestamp"][:10] <= str(day)], day, reference, svc.c.embedder, cfg, g, scope_thr, env)
            nightly.append({"as_of": str(day), "day": d, "fired": rep["fired"]})
            for det in rep["fired"]:
                first_night.setdefault(det, d)
            alerts += [{**a, "day": d} for a in rep["alerts"] if a["detector"] not in {x["detector"] for x in alerts}]
        fired = {d for d, night in first_night.items() if night >= ONSET[scenario]}
        early = sorted({det for n in nightly if n["day"] < ONSET[scenario] for det in n["fired"]})
        entry = {"fired": sorted(fired), "first_night": first_night, "expected": sorted(EXPECTED[scenario]),
                 "onset_day": ONSET[scenario], "false_alarms_before_onset": early,
                 "missing": sorted(EXPECTED[scenario] - fired), "requests": info["requests"], "alerts": alerts,
                 "nightly_runs": nightly}
        if scenario == "baseline":
            entry["nightly_false_alarms"] = sum(bool(n["fired"]) for n in nightly)
        results[scenario] = entry
        series[scenario] = {"daily": rep["daily"], "golden": [{"day": x.get("simulated_day"), **x["metrics"]} for x in golden]}
        (out_reports / f"{scenario}.json").write_text(json.dumps(rep, indent=1))
        print(f"{scenario:24s} fired={sorted(fired)} expected={sorted(EXPECTED[scenario])} ({time.perf_counter() - t:.0f}s)", flush=True)

    ok = all(not r["missing"] and not r["false_alarms_before_onset"] for r in results.values())
    summary = {"ok": ok, "reference": reference.description, "per_day": args.per_day, "days": DAYS,
               "embedder": svc.c.embedder.name, "results": results}
    (ROOT / "monitor" / "validation" / "results.json").write_text(json.dumps(summary, indent=1))
    write_markdown(summary)
    plot(series)
    print("detector validation:", "PASS" if ok else "FAIL", f"({time.perf_counter() - t0:.0f}s)")
    return 0 if ok else 1


def write_markdown(summary: dict) -> None:
    res = summary["results"]
    lines = ["# Drift detector validation", "",
             f"Simulated traffic: {summary['days']} days x {summary['per_day']} requests/day per scenario (days 0-6 baseline, "
             f"scenario from day 7; model drift degrades the model from day 9), nightly job run on each of days 7-13 "
             f"as the cron would; cells show the first night a detector fired. Reference: {summary['reference']}. "
             f"Embedder: `{summary['embedder']}`.", "",
             "| Scenario | Onset | " + " | ".join(DETECTORS) + " | False alarms before onset | Pass |",
             "|---|---|" + "---|" * (len(DETECTORS) + 2)]
    for s, r in res.items():
        cells = []
        for d in DETECTORS:
            if d in r["fired"]:
                night = min(n["day"] for n in r["nightly_runs"] if d in n["fired"] and n["day"] >= r["onset_day"])
                cells.append(f"**fired** (day {night})" if d in r["expected"] else f"fired (day {night})")
            else:
                cells.append("MISSED" if d in r["expected"] else "·")
        ok = "yes" if not r["missing"] and not r["false_alarms_before_onset"] else "**no**"
        early = ", ".join(r["false_alarms_before_onset"]) or "none"
        onset = "–" if r["onset_day"] > DAYS else f"day {r['onset_day']}"
        lines.append(f"| {s} | {onset} | " + " | ".join(cells) + f" | {early} | {ok} |")
    b = res.get("baseline")
    if b:
        lines += ["", f"Baseline: **{b['nightly_false_alarms']}** false alarms over {len(b['nightly_runs'])} consecutive nightly runs."]
    lines += ["", "**fired** = expected detector fired; fired = additional detector fired (secondary signal); "
              "MISSED = expected detector stayed quiet; · = quiet.", ""]
    for s, r in res.items():
        if r["alerts"]:
            lines.append(f"- **{s}**: " + "; ".join(f"{a['detector']}/{a['metric']}={a['value']}" for a in r["alerts"][:6]))
    (ROOT / "docs" / "drift_validation.md").write_text("\n".join(lines) + "\n")


def plot(series: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("out_of_scope_rate", "topic_shift", "Out-of-scope rate (topic shift)"),
              ("refusal_rate", "hallucination_pressure", "Refusal rate (hallucination pressure)"),
              ("prompt_injection", "attack_wave", "Injection trigger rate (attack wave)"),
              ("reference_number_recall", "model_drift", "Golden: reference-figure recall (model drift)")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.4))
    for ax, (metric, scen, title) in zip(axes, panels, strict=True):
        for s, color in ((scen, "#dc3912"), ("baseline", "#3366cc")):
            if s not in series:
                continue
            if metric == "reference_number_recall":
                pts = [(g["day"], g[metric]) for g in series[s]["golden"]]
            elif metric == "prompt_injection":
                pts = [(i, d["triggers"][metric] / d["n"]) for i, d in enumerate(series[s]["daily"])]
            else:
                pts = [(i, d[metric]) for i, d in enumerate(series[s]["daily"])]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", ms=3, color=color, label=s)
        ax.axvline(6.5, color="grey", ls="--", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("simulated day")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "assets" / "drift_scenarios.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
