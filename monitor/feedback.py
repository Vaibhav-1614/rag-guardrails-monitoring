"""Feedback loop (design doc §9): flagged production traces become new eval cases.

    python -m monitor.feedback export --env production --days 1      # -> monitoring/review_queue.jsonl
    # review: set "label" on each line to true_issue | false_positive | new_question_type | skip
    python -m monitor.feedback apply monitoring/review_queue.jsonl  # -> eval/golden.jsonl, eval/redteam/*.jsonl
    python -m monitor.feedback stats                                 # eval-set growth from production

A trace is flagged when a guard failed (in either mode), groundedness was low, or a user gave a
thumbs-down. ``--langfuse-queue`` also pushes flagged traces into a Langfuse annotation queue.

Labels:
- true_issue        -> appended to the matching red-team set (the guards must keep catching it)
- false_positive    -> appended to the golden set (the guards must let it through)
- new_question_type -> appended to the golden set with the reviewer's reference answer
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

from rag.companies import get_registry
from rag.config import load_settings, resolve

SET_FOR_GUARD = {
    "prompt_injection": "injection", "charset": "injection", "context_sanitizer": "indirect",
    "scope": "scope_advice", "advice": "scope_advice", "advice_output": "scope_advice",
    "numeric_grounding": "hallucination_bait", "nli_grounding": "hallucination_bait", "citation": "hallucination_bait",
    "retrieval_conf": "golden",
}


def flag_reasons(rec: dict, feedback: dict[str, float]) -> list[str]:
    md, sc = rec.get("metadata") or {}, rec.get("scores") or {}
    reasons = [f"guard:{g}" for g in md.get("guard_trigger") or [] if g not in ("pii", "pii_output")]
    if sc.get("groundedness") is not None and sc["groundedness"] < 0.8:
        reasons.append("low_groundedness")
    if feedback.get(rec.get("id")) == 0:
        reasons.append("thumbs_down")
    if md.get("status") == "error":
        reasons.append("error")
    return reasons


def suggested_set(reasons: list[str]) -> str:
    for r in reasons:
        if r.startswith("guard:"):
            return SET_FOR_GUARD.get(r[6:], "golden")
    return "golden"


def export(records: list[dict], feedback: dict[str, float], out: Path, emerging: list[str] | None = None) -> list[dict]:
    emerging_set = {e.strip().lower() for e in emerging or []}
    items = []
    for rec in records:
        reasons = flag_reasons(rec, feedback)
        q = (rec.get("input") or {}).get("question", "")
        if q.strip().lower() in emerging_set:
            reasons.append("emerging_topic")
        if not reasons:
            continue
        items.append({"trace_id": rec["id"], "timestamp": rec["timestamp"], "environment": rec.get("environment"),
                      "question": q, "answer": (rec.get("output") or {}).get("answer", ""), "reasons": reasons,
                      "suggested_set": suggested_set(reasons), "label": None, "reference_answer": None, "notes": ""})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    return items


def push_langfuse_queue(items: list[dict], queue_id: str, transport: httpx.BaseTransport | None = None) -> int:
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
    auth = (os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])
    n = 0
    with httpx.Client(base_url=host, auth=auth, timeout=30, transport=transport) as c:
        for it in items:
            r = c.post(f"/api/public/annotation-queues/{queue_id}/items", json={"objectId": it["trace_id"], "objectType": "TRACE"})
            n += r.status_code < 300
    return n


def _append(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _existing_sources(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line).get("source", "") for line in path.read_text().splitlines() if line.strip()}


def apply(queue: Path, eval_dir: Path) -> Counter:
    reg = get_registry()
    counts: Counter = Counter()
    for line in queue.read_text().splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        label = it.get("label")
        if label in (None, "", "skip"):
            counts["skipped"] += 1
            continue
        src = f"production:{it['trace_id']}"
        short = hashlib.sha1(it["trace_id"].encode()).hexdigest()[:8]
        tickers = list(reg.detect(it["question"]).covered)
        if label in ("false_positive", "new_question_type"):
            path = eval_dir / "golden.jsonl"
            if src in _existing_sources(path):
                continue
            _append(path, {"id": f"gp_{short}", "question": it["question"], "ticker": tickers[0] if tickers else None,
                           "answerable": True, "type": "production", "reference_answer": it.get("reference_answer"),
                           "source": src})
            counts[f"golden:{label}"] += 1
        elif label == "true_issue":
            target = it.get("suggested_set", "golden")
            path = eval_dir / "redteam" / f"{target}.jsonl"
            if src in _existing_sources(path):
                continue
            if target == "injection":
                row = {"id": f"inj_p_{short}", "text": it["question"], "label": "attack", "category": "production", "split": "test"}
            elif target == "scope_advice":
                guard = next((r[6:] for r in it["reasons"] if r.startswith("guard:")), "scope")
                cat, exp = ("advice", "refuse_advice") if guard.startswith("advice") else ("off_topic", "refuse_scope")
                row = {"id": f"sa_p_{short}", "text": it["question"], "category": cat, "expected": exp, "split": "test"}
            elif target == "hallucination_bait":
                row = {"id": f"hb_p_{short}", "question": it["question"], "ticker": tickers[0] if tickers else None,
                       "category": "production", "expected": "refuse_or_no_unsupported_numbers", "split": "test"}
            else:
                counts["unsupported_set"] += 1
                continue
            _append(path, {**row, "source": src, "notes": it.get("notes", "")})
            counts[f"{target}:true_issue"] += 1
    return counts


def stats(eval_dir: Path) -> dict:
    out = {}
    for p in [eval_dir / "golden.jsonl", *sorted((eval_dir / "redteam").glob("*.jsonl"))]:
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        out[p.stem] = {"total": len(rows), "from_production": sum(str(r.get("source", "")).startswith("production:") for r in rows)}
    out["from_production_total"] = sum(v["from_production"] for v in out.values() if isinstance(v, dict))
    return out


def main() -> None:
    from rag.tracing import make_backend

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--env", default="production")
    e.add_argument("--days", type=int, default=1)
    e.add_argument("--source", default="auto")
    e.add_argument("--out", default="monitoring/review_queue.jsonl")
    e.add_argument("--report", default="monitoring/reports/latest.json", help="emerging-topic examples are flagged too")
    e.add_argument("--langfuse-queue", default=None)
    a = sub.add_parser("apply")
    a.add_argument("queue")
    sub.add_parser("stats")
    args = ap.parse_args()
    eval_dir = resolve("eval")
    if args.cmd == "export":
        backend = make_backend(load_settings(), None if args.source == "auto" else args.source)
        end = datetime.now(UTC)
        records = backend.read(end - timedelta(days=args.days), end, args.env)
        feedback = backend.read_feedback(end - timedelta(days=args.days), end, args.env) if hasattr(backend, "read_feedback") else {}
        emerging = []
        rp = resolve(args.report)
        if rp.exists():
            rep = json.loads(rp.read_text())
            emerging = [ex for c in rep.get("detectors", {}).get("emerging_topics", {}).get("clusters", []) for ex in c.get("examples", [])]
        items = export(records, feedback, resolve(args.out), emerging)
        print(f"{len(items)} of {len(records)} traces flagged -> {args.out}")
        if args.langfuse_queue:
            print("pushed to Langfuse annotation queue:", push_langfuse_queue(items, args.langfuse_queue))
    elif args.cmd == "apply":
        print(dict(apply(Path(args.queue), eval_dir)))
    else:
        print(json.dumps(stats(eval_dir), indent=2))


if __name__ == "__main__":
    main()
