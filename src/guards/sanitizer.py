"""§6.2 Context sanitizer (T2): neutralise instruction-like text inside retrieved chunks.

Delimiting is done by the prompt builder (``<chunk id=.. ticker=.. section=..>``, with ``<``/``>``
escaped so chunk text can't forge a tag), and the system prompt says chunk content is data, not
instructions. This guard adds the third layer: it runs the injection rules over every chunk and,
replaces offending sentences (``strip: sentence``) or the whole chunk (``strip: chunk``, quarantine)
with a placeholder before the LLM sees them.
"""
from __future__ import annotations

from typing import Any

from guards.base import BaseGuard, GuardContext
from guards.injection import rule_scan
from rag.text import split_sentences

PLACEHOLDER = "[removed: instruction-like text]"


class ContextSanitizer(BaseGuard):
    name = "context_sanitizer"
    stage = "context"
    fail_action = "redact"

    def sanitize(self, text: str, threshold: float) -> tuple[str, list[dict]]:
        hits: list[dict] = []
        out_lines = []
        for line in text.splitlines():
            sents = split_sentences(line) or [line]
            kept = []
            for s in sents:
                score, matches = rule_scan(s)
                if score >= threshold:
                    hits.append({"score": round(score, 3), "rules": [m.rule for m in matches][:4], "snippet": s[:120]})
                    kept.append(PLACEHOLDER)
                else:
                    kept.append(s)
            out_lines.append(" ".join(kept))
        return "\n".join(out_lines), hits

    def evaluate(self, ctx: GuardContext) -> tuple[bool, float | None, dict[str, Any]]:
        threshold = float(self.config.get("threshold", 0.8))
        per_chunk = []
        sanitized = []
        for i, c in enumerate(ctx.chunks, 1):
            score, _ = rule_scan(c["text"])
            if score < threshold:  # fast path: nothing suspicious anywhere in the chunk
                sanitized.append(c)
                continue
            clean, hits = self.sanitize(c["text"], threshold)
            if hits and self.config.get("strip", "sentence") == "chunk":
                # Quarantine: once a chunk carries an injected instruction, none of it is trusted.
                clean = PLACEHOLDER
            if hits:
                per_chunk.append({"chunk": i, "chunk_id": c.get("chunk_id"), "hits": hits})
            sanitized.append({**c, "text": clean})
        n_hits = sum(len(p["hits"]) for p in per_chunk)
        ctx.cache["sanitized_chunks"] = sanitized
        return not per_chunk, float(n_hits), {"flagged_chunks": per_chunk, "n_hits": n_hits}
