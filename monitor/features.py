"""Turn trace records (local JSONL or Langfuse) into per-request rows and daily aggregates."""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import numpy as np

OUTPUT_METRICS = ["refusal_rate", "regeneration_rate", "flagged_rate", "unsupported_rate", "citations_mean",
                  "answer_chars_mean", "groundedness_mean"]
OPS_METRICS = ["latency_p95_ms", "error_rate", "fallback_rate", "tokens_per_request"]


def parse_ts(ts: Any) -> datetime:
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=UTC)
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


def to_row(rec: dict) -> dict:
    md = rec.get("metadata") or {}
    sc = rec.get("scores") or {}
    lat = md.get("latency_ms") or {}
    ts = parse_ts(rec["timestamp"])
    status = md.get("status", "answered")
    return {
        "id": rec.get("id"),
        "ts": ts,
        "day": ts.astimezone(UTC).strftime("%Y-%m-%d"),
        "env": rec.get("environment"),
        "question": (rec.get("input") or {}).get("question", ""),
        "status": status,
        "refusal_reason": md.get("refusal_reason"),
        "scope": sc.get("scope"),
        "scope_reason": md.get("scope_reason"),
        "retrieval_top": sc.get("retrieval_top"),
        "groundedness": sc.get("groundedness"),
        "numeric_support": sc.get("numeric_support"),
        "answer_chars": md.get("answer_chars", 0),
        "n_citations": md.get("n_citations", 0),
        "unsupported": md.get("unsupported_numbers") or 0,
        "regenerated": bool(md.get("regenerated")),
        "flags": md.get("flags") or [],
        "triggers": md.get("guard_trigger") or [],
        "latency_total": lat.get("total"),
        "tokens": (lat.get("tokens_in") or 0) + (lat.get("tokens_out") or 0),
        "model": md.get("model"),
        "config_version": md.get("config_version"),
    }


def _mean(vals: list[float]) -> float:
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def daily_aggregates(rows: list[dict], guards: list[str]) -> list[dict]:
    by_day: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_day[r["day"]].append(r)
    out = []
    for day in sorted(by_day):
        rs = by_day[day]
        n = len(rs)
        answered = [r for r in rs if r["status"] == "answered"]
        lat = [r["latency_total"] for r in rs if r["latency_total"] is not None]
        agg = {
            "day": day,
            "n": n,
            "refusal_rate": sum(r["status"] in ("refused", "fallback") for r in rs) / n,
            "out_of_scope_rate": sum(r["scope_reason"] is not None for r in rs) / n,
            "low_confidence_rate": sum(r["refusal_reason"] == "retrieval_conf" for r in rs) / n,
            "regeneration_rate": sum(r["regenerated"] for r in rs) / n,
            "flagged_rate": sum(any(f.startswith(("unverified:", "flag:")) for f in r["flags"]) for r in rs) / n,
            "unsupported_rate": (sum(r["unsupported"] > 0 for r in answered) / len(answered)) if answered else float("nan"),
            "citations_mean": _mean([r["n_citations"] for r in answered]),
            "answer_chars_mean": _mean([r["answer_chars"] for r in answered]),
            "groundedness_mean": _mean([r["groundedness"] for r in answered]),
            "scope_mean": _mean([r["scope"] for r in rs]),
            "retrieval_top_mean": _mean([r["retrieval_top"] for r in rs]),
            "latency_p95_ms": float(np.percentile(lat, 95)) if lat else float("nan"),
            "error_rate": sum(r["status"] == "error" for r in rs) / n,
            "fallback_rate": sum(r["status"] == "fallback" for r in rs) / n,
            "tokens_per_request": _mean([r["tokens"] for r in rs]),
            "triggers": {g: sum(g in r["triggers"] for r in rs) for g in guards},
        }
        out.append(agg)
    return out
