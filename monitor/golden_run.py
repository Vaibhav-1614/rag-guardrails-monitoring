"""Nightly golden-set run against production (upstream model drift, D2).

Runs independently of PRs. If golden scores move while the code/config version is unchanged, the
provider changed something. Results are appended to a JSONL history that the drift job reads.

    python -m monitor.golden_run --target https://your-service.example.com   # deployed API
    python -m monitor.golden_run --target inprocess                          # local service
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from guards.numeric import NumericChecker
from rag.config import resolve


def load_golden(limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in resolve("eval/golden.jsonl").read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["answerable"]]
    if limit:
        numeric = [r for r in rows if r["type"] == "numeric"]
        prose = [r for r in rows if r["type"] != "numeric"]
        rows = (numeric + prose)[:limit]
    return rows


def code_version() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"][:12]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def reference_numbers_found(answer: str, ref_numbers: list[str], ticker: str) -> bool:
    """Does the answer state every reference figure (allowing re-scaling such as 416,161 million -> $416.2 billion)?"""
    ck = NumericChecker(rel_tol=0.0005)
    for num in ref_numbers:
        checks = ck.check(f"The figure is {num} [1].", [{"text": answer, "ticker": ticker}], table_scale={ticker: 1e6})
        if not checks or any(c.status not in ("supported", "derived") for c in checks):
            return False
    return True


def _guard(resp: dict, name: str) -> dict | None:
    gs = [g for g in resp.get("guards", []) if g["guard"] == name]
    return gs[-1] if gs else None


def summarize(items: list[dict], responses: list[dict]) -> dict[str, Any]:
    answered = [r for r in responses if r["status"] == "answered"]
    numeric_items = [(i, r) for i, r in zip(items, responses, strict=True) if i["type"] == "numeric"]
    found = [reference_numbers_found(r["answer"], i["reference_numbers"], i["ticker"]) for i, r in numeric_items]

    def mean_score(name: str) -> float:
        vals = [g["score"] for r in answered if (g := _guard(r, name)) and g.get("score") is not None]
        return float(np.mean(vals)) if vals else float("nan")

    unsupported = [r for r in answered if (g := _guard(r, "numeric_grounding")) and g["detail"].get("unsupported")]
    return {
        "n": len(responses),
        "answered_rate": round(len(answered) / max(len(responses), 1), 4),
        "reference_number_recall": round(float(np.mean(found)), 4) if found else float("nan"),
        "citation_coverage_mean": round(mean_score("citation"), 4),
        "numeric_support_mean": round(mean_score("numeric_grounding"), 4),
        "groundedness_mean": round(mean_score("nli_grounding"), 4),
        "unsupported_answer_rate": round(len(unsupported) / max(len(answered), 1), 4),
        "answer_chars_mean": round(float(np.mean([len(r["answer"]) for r in answered])), 1) if answered else float("nan"),
        "regeneration_rate": round(float(np.mean([bool(r.get("regenerated")) for r in responses])), 4),
    }


def run(target: str = "inprocess", service=None, limit: int | None = None, timeout: float = 120) -> dict[str, Any]:
    items = load_golden(limit)
    responses: list[dict] = []
    if target == "inprocess":
        if service is None:
            from rag.service import RAGService

            service = RAGService.build(trace_backend="none")
        for it in items:
            responses.append(service.ask(it["question"], trace=False).model_dump(mode="json"))
        meta = {"provider": service.c.generator.provider, "model": service.c.generator.model, "config": service.c.pipeline.version}
    else:
        with httpx.Client(base_url=target.rstrip("/"), timeout=timeout) as client:
            for it in items:
                r = client.post("/ask", json={"question": it["question"]}, headers={"X-Trace-Env": "golden-nightly"})
                r.raise_for_status()
                responses.append(r.json())
            health = client.get("/health").json()
        meta = {"provider": health.get("provider"), "model": None, "config": health.get("guards_config")}
        meta["model"] = next((r.get("model") for r in responses if r.get("model")), None)
    return {"timestamp": datetime.now(UTC).isoformat(), "target": target, "code_version": code_version(), **meta,
            "metrics": summarize(items, responses)}


def append_history(result: dict, path: str | Path) -> Path:
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    return p


def load_history(path: str | Path) -> list[dict]:
    p = resolve(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=os.environ.get("MONITOR_TARGET_URL", "inprocess"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--history", default="monitoring/golden_history.jsonl")
    args = ap.parse_args()
    res = run(args.target, limit=args.limit)
    append_history(res, args.history)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
