"""§6.5 NLI groundedness (T6, general claims).

Each claim sentence is a hypothesis; its cited chunks are premises. A sentence is grounded when
P(entailment) >= threshold for at least one cited chunk; the answer score is the grounded fraction.

Backends:
- ``cross-encoder``: a local NLI cross-encoder (default ``cross-encoder/nli-deberta-v3-xsmall``,
  ~70 MB, CPU). Premises are split into sentence windows so they fit the model's context.
- ``lexical``: fallback when the model can't be loaded. "Entailment" is approximated by the share of
  the sentence's content words (stemmed) that appear in the cited chunk window. It is a weaker,
  recall-oriented proxy; reports always state which backend produced a number.

NLI models handle tables and numbers poorly; numeric grounding (§6.4) covers that gap.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from functools import lru_cache
from typing import Any

import numpy as np

from guards.base import BaseGuard, GuardContext
from guards.citation import cited_ids, claim_sentences, strip_citations
from rag.prompts import NOT_FOUND
from rag.text import content_terms, split_sentences, stem

logger = logging.getLogger(__name__)


class CrossEncoderNLI:
    name = "cross-encoder"

    def __init__(self, model: str) -> None:
        from sentence_transformers import CrossEncoder  # optional dependency

        self.model_name = model
        self._model = CrossEncoder(model, device="cpu")
        labels = getattr(self._model.model.config, "id2label", {}) or {}
        self._entail_idx = next((int(i) for i, lbl in labels.items() if "entail" in str(lbl).lower()), 1)
        self._lock = threading.Lock()

    def entailment(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0)
        with self._lock:
            logits = np.asarray(self._model.predict(pairs, apply_softmax=False, show_progress_bar=False))
        if logits.ndim == 1:
            return 1 / (1 + np.exp(-logits))
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return (e / e.sum(axis=1, keepdims=True))[:, self._entail_idx]


class LexicalNLI:
    name = "lexical"
    model_name = "lexical-overlap-v1"

    def entailment(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        out = []
        for premise, hyp in pairs:
            h = {stem(t) for t in content_terms(hyp)}
            p = {stem(t) for t in content_terms(premise)}
            out.append(len(h & p) / len(h) if h else 1.0)
        return np.asarray(out, dtype=float)


@lru_cache(maxsize=2)
def load_nli(model: str):
    try:
        return CrossEncoderNLI(model)
    except Exception as e:
        logger.warning("NLI model %r unavailable (%s); using lexical fallback", model, type(e).__name__)
        return LexicalNLI()


def _windows(text: str, size: int = 4, stride: int = 2) -> list[str]:
    sents = split_sentences(text)
    if len(sents) <= size:
        return [" ".join(sents)] if sents else [text]
    return [" ".join(sents[i:i + size]) for i in range(0, len(sents) - size + stride, stride)]


class NLIGroundingGuard(BaseGuard):
    name = "nli_grounding"
    stage = "output"
    fail_action = "flag"

    def __init__(self, config: dict | None = None, backend=None) -> None:
        super().__init__(config)
        want = self.config.get("backend", "auto")
        if backend is not None:
            self.backend = backend
        elif want == "lexical":
            self.backend = LexicalNLI()
        else:
            self.backend = load_nli(self.config.get("model", "cross-encoder/nli-deberta-v3-xsmall"))
        if self.backend.name == "lexical":
            self.threshold = float(self.config.get("lexical_threshold", 0.6))
        else:
            self.threshold = float(self.config.get("threshold", 0.7))
        self.min_fraction = float(self.config.get("min_grounded_fraction", 0.8))
        self.sample_rate = float(self.config.get("sample_rate", 1.0))

    def sampled(self, ctx: GuardContext) -> bool:
        if self.sample_rate >= 1.0:
            return True
        key = (ctx.trace_id or ctx.question).encode()
        return int(hashlib.sha1(key).hexdigest()[:8], 16) / 0xFFFFFFFF < self.sample_rate

    def evaluate(self, ctx: GuardContext) -> tuple[bool, float | None, dict[str, Any]]:
        answer = (ctx.answer or "").strip()
        if not answer or answer.startswith(NOT_FOUND[:30]):
            return True, 1.0, {"refusal": True, "backend": self.backend.name}
        if not self.sampled(ctx):
            return True, None, {"sampled": False, "backend": self.backend.name}
        claims = claim_sentences(answer)
        if not claims:
            return True, 1.0, {"n_claims": 0, "backend": self.backend.name}
        pairs, owners = [], []
        for si, s in enumerate(claims):
            ids = [i for i in cited_ids(s) if 1 <= i <= len(ctx.chunks)] or list(range(1, len(ctx.chunks) + 1))
            hyp = strip_citations(s)
            for i in ids:
                for w in _windows(ctx.chunks[i - 1]["text"]):
                    pairs.append((w, hyp))
                    owners.append(si)
        probs = self.backend.entailment(pairs)
        best = np.zeros(len(claims))
        for o, p in zip(owners, probs, strict=True):
            best[o] = max(best[o], float(p))
        grounded = best >= self.threshold
        frac = float(grounded.mean())
        detail = {
            "backend": self.backend.name,
            "model": getattr(self.backend, "model_name", None),
            "threshold": self.threshold,
            "n_claims": len(claims),
            "n_pairs": len(pairs),
            "sentence_scores": [round(float(b), 3) for b in best],
            "ungrounded": [claims[i][:160] for i in np.where(~grounded)[0]],
        }
        return frac >= self.min_fraction, frac, detail
