"""§6.1 Retrieval-confidence gate (T8): don't let the model answer from irrelevant context."""
from __future__ import annotations

from typing import Any

from guards.base import BaseGuard, GuardContext


class RetrievalConfidenceGuard(BaseGuard):
    name = "retrieval_conf"
    stage = "retrieval"
    fail_action = "block"

    def evaluate(self, ctx: GuardContext) -> tuple[bool, float | None, dict[str, Any]]:
        r = ctx.retrieval
        min_top = float(self.config.get("min_top_score", 0.45))
        floor = float(self.config.get("relevance_floor", 0.30))
        min_chunks = int(self.config.get("min_chunks", 1))
        if r is None or not r.chunks:
            return False, 0.0, {"reason": "no_chunks"}
        top = r.top_score
        n_above = r.n_above(floor)
        reasons = []
        if top < min_top:
            reasons.append("low_top_score")
        if n_above < min_chunks:
            reasons.append("few_relevant_chunks")
        detail = {"top_score": round(top, 4), "n_above_floor": n_above, "min_top_score": min_top,
                  "relevance_floor": floor, "reasons": reasons,
                  "scores": [round(c.relevance, 3) for c in r.chunks]}
        return not reasons, float(top), detail
