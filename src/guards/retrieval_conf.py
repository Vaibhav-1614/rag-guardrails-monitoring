"""§6.1 Retrieval-confidence gate (T8): don't let the model answer when the filings can't.

The decision comes from the upstream service's answerability check, which runs on what was
actually retrieved: market data, forecasts, periods outside the filings, entities the corpus never
mentions, and low evidence (the re-ranker's score for the best passage, or term coverage without
a re-ranker). On top of that:

- ``facts_narrowing_check``: refuse when the facts table (company-wide totals) is about to answer a
  question about a segment, business line or per-unit ratio; the figure would be real but wrong.
- ``min_confidence``: an optional stricter floor, calibrated by the eval.
"""
from __future__ import annotations

import re
from typing import Any

from guards.base import BaseGuard, GuardContext

# The facts table holds company-wide totals only. A question narrowed to a part of the company or to
# a ratio must not be answered with the total ("revenue of Apple's automotive segment" -> total revenue).
_NARROWED = re.compile(
    r"\b(?:segment|division|business unit|subsidiar(?:y|ies)|product line)s?\b"
    r"|\bper (?:employee|customer|user|person|store|vehicle|barrel|query|subscriber|unit|square foot)\b"
    r"|\b(?:from|of|in|by|at) (?:its|their|the|his|her) (?:[\w&-]+ ){1,4}?(?:business|operations?|desk|exchange|arm|unit|platform|franchise)\b"
    r"|['’]s (?:[\w&-]+ ){1,4}?(?:business|desk|exchange|arm|unit|franchise)\b",
    re.I,
)


def narrowed_facts_question(question: str) -> str | None:
    m = _NARROWED.search(question)
    return m.group(0) if m else None


class RetrievalConfidenceGuard(BaseGuard):
    name = "retrieval_conf"
    stage = "retrieval"
    fail_action = "block"

    def evaluate(self, ctx: GuardContext) -> tuple[bool, float | None, dict[str, Any]]:
        r = ctx.retrieval
        if r is None or not r.chunks:
            return False, 0.0, {"reason": "no_chunks", "reasons": ["no_chunks"]}
        reasons = []
        if not r.answerable:
            reasons.append(r.reason)
        if r.facts_mode != "none" and self.config.get("facts_narrowing_check", True):
            hit = narrowed_facts_question(ctx.effective_question)
            if hit:
                reasons.append("facts_scope_mismatch")
        floor = self.config.get("min_confidence")
        if floor is not None and r.facts_mode == "none" and r.confidence < float(floor):
            reasons.append("below_min_confidence")
        detail = {"confidence": round(r.confidence, 4), "coverage": round(r.coverage, 4),
                  "evidence_score": r.evidence_score, "upstream_reason": r.reason, "facts_mode": r.facts_mode,
                  "retrieval_version": r.version, "reasons": reasons}
        return not reasons, float(r.confidence), detail
