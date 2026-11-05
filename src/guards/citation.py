"""§6.3 Citation validator (T7), deterministic.

- every [n] must refer to a chunk that was actually provided;
- every sentence that states a fact (contains a number or a named entity) must carry a citation.

Whether the cited chunk *supports* the sentence is checked by numeric grounding (§6.4) and NLI
(§6.5); this guard also records a cheap lexical-overlap signal per cited sentence for analysis.
"""
from __future__ import annotations

import re
from typing import Any

from guards.base import BaseGuard, GuardContext
from rag.prompts import NOT_FOUND
from rag.text import content_terms, split_sentences, stem

CITE = re.compile(r"\[(\d+(?:\s*[,\-–]\s*\d+)*)\]")
_HAS_NUMBER = re.compile(r"\d")
_FILLER = re.compile(r"^(?:in summary|overall|in short|however|additionally|note|according to the (?:10-k|filing))\b", re.I)


def cited_ids(sentence: str) -> list[int]:
    ids: list[int] = []
    for grp in CITE.findall(sentence):
        for part in re.split(r"\s*,\s*", grp):
            if re.search(r"[\-–]", part):
                a, b = (int(x) for x in re.split(r"\s*[\-–]\s*", part)[:2])
                ids.extend(range(min(a, b), max(a, b) + 1))
            else:
                ids.append(int(part))
    return ids


def strip_citations(text: str) -> str:
    return re.sub(r"\s*" + CITE.pattern, "", text)


_FRAMING = re.compile(r"^(?:here(?:'s| is| are)|based on (?:the|these) (?:chunks|filings?|excerpts)|in summary|to summarize|overall|in short|"
                      r"note that|i (?:couldn't|could not|cannot|can't)|the (?:filing|10-k|chunks?) (?:does|do) not)\b", re.I)


def is_claim(sentence: str, strict: bool = True) -> bool:
    """Does this sentence state something that needs a citation?

    strict (default): every substantive sentence (>= 6 words) except framing/hedging lines.
    heuristic (doc §6.3): only sentences with a number or a named entity.
    """
    body = strip_citations(sentence).strip()
    if len(body) < 12 or body == NOT_FOUND.rstrip(".") or body.endswith("?"):
        return False
    if strict:
        return len(body.split()) >= 6 and not _FRAMING.match(body)
    if _HAS_NUMBER.search(body):
        return True
    # A capitalised word that does not start the sentence ~ a named entity.
    words = body.split()
    return any(w[:1].isupper() and i > 0 and not w.isupper() or (w.isupper() and len(w) > 1 and i > 0) for i, w in enumerate(words))


def claim_sentences(answer: str, strict: bool = True) -> list[str]:
    return [s for s in split_sentences(answer) if is_claim(s, strict) and not _FILLER.match(s.strip())]


def strip_uncited(answer: str, strict: bool = True) -> str:
    """Fallback: drop claim sentences that carry no citation."""
    kept = [s for s in split_sentences(answer) if not (is_claim(s, strict) and not cited_ids(s))]
    return " ".join(kept).strip()


class CitationGuard(BaseGuard):
    name = "citation"
    stage = "output"
    fail_action = "regenerate"

    def evaluate(self, ctx: GuardContext) -> tuple[bool, float | None, dict[str, Any]]:
        answer = (ctx.answer or "").strip()
        n_chunks = len(ctx.chunks)
        if not answer or answer.startswith(NOT_FOUND[:30]):
            return True, 1.0, {"refusal": True}
        strict = self.config.get("claims", "strict") == "strict"
        sentences = split_sentences(answer)
        invalid = sorted({i for s in sentences for i in cited_ids(s) if not 1 <= i <= n_chunks})
        claims = [s for s in sentences if is_claim(s, strict) and not _FILLER.match(s.strip())]
        uncited = [s for s in claims if not cited_ids(s)]
        low_overlap = []
        for s in claims:
            ids = [i for i in cited_ids(s) if 1 <= i <= n_chunks]
            if not ids:
                continue
            words = {stem(t) for t in content_terms(strip_citations(s))}
            ctx_words = {stem(t) for i in ids for t in content_terms(ctx.chunks[i - 1]["text"])}
            if words and len(words & ctx_words) / len(words) < 0.2:
                low_overlap.append(s[:160])
        n_cited = sum(1 for s in claims if cited_ids(s))
        score = n_cited / len(claims) if claims else 1.0
        detail = {
            "n_sentences": len(sentences),
            "n_claims": len(claims),
            "n_citations": sum(len(cited_ids(s)) for s in sentences),
            "invalid_citations": invalid,
            "uncited_claims": [s[:160] for s in uncited],
            "low_overlap_citations": low_overlap,
        }
        reasons = []
        if invalid:
            reasons.append(f"citations {invalid} refer to chunks that were not provided (valid: 1-{n_chunks})")
        if uncited:
            reasons.append(f"{len(uncited)} factual sentence(s) have no citation")
        detail["reasons"] = reasons
        return not reasons, float(score) if not invalid else 0.0, detail
