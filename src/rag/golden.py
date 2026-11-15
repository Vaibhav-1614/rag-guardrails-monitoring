"""The golden set: upstream's eval/golden.jsonl (176 questions with dev/test splits), plus this
repo's eval/golden_extra.jsonl, where the feedback loop appends questions from production."""
from __future__ import annotations

import json
import re

from rag.config import resolve
from rag.upstream import upstream_root


def _read(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_golden(split: str = "all", settings: dict | None = None) -> list[dict]:
    rows = []
    for r in _read(upstream_root(settings) / "eval" / "golden.jsonl"):
        tickers = r.get("tickers") or []
        rows.append({
            "id": r["id"], "question": r["question"], "answerable": bool(r["answerable"]), "type": r["type"],
            "split": r.get("split", "dev"), "tickers": tickers, "ticker": tickers[0] if tickers else None,
            "category": r.get("category"), "answer_contains": r.get("answer_contains"),
            "evidence_any": r.get("evidence_any"), "reference_answer": r.get("reference_answer"),
            "source": "rag-eval-platform/eval/golden.jsonl",
        })
    for r in _read(resolve("eval/golden_extra.jsonl")):
        rows.append({"split": "dev", "tickers": [r["ticker"]] if r.get("ticker") else [], "category": None,
                     "answer_contains": None, "evidence_any": None, **r})
    return [r for r in rows if split == "all" or r["split"] == split]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


def answer_check(answer: str, item: dict) -> bool | None:
    """Upstream's no-LLM answer check: every group in `answer_contains` has one variant in the answer."""
    groups = item.get("answer_contains")
    if not groups:
        return None
    a = _norm(answer)
    return all(any(_norm(v) in a for v in grp) for grp in groups)
