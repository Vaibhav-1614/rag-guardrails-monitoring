"""§5.4 Scope check: is this a question about the covered companies' 10-K filings?

score = w_domain * domain + w_similarity * similarity + w_company * mentions_covered_company

- domain: for each content word, the log-likelihood ratio of its frequency in the 10-K corpus vs.
  general English (``wordfreq``), squashed with a sigmoid and averaged. "revenue", "impairment" score
  high; "poem", "recipe" score low. Words unseen in both are ignored.
- similarity: max cosine between the query embedding and the (ticker, section) centroids of the
  corpus, rescaled to [0, 1] with per-embedder floor/ceiling. Reuses the retrieval embedder.
- unknown company: a question naming a company outside the corpus (and none inside it) fails with a
  specific refusal, whatever its score.

The score is also the main input to query/scope drift monitoring.
"""
from __future__ import annotations

import math
import re
from functools import cached_property
from typing import Any

import numpy as np

from guards.base import BaseGuard, GuardContext
from rag.companies import CompanyRegistry
from rag.corpus import Corpus
from rag.embedder import Embedder
from rag.text import STOPWORDS

try:
    from wordfreq import word_frequency
except ImportError:  # pragma: no cover
    word_frequency = None

_QUESTION_WORDS = frozenset("""what which how many much does did do is are was were who whom whose when where why
    can could would should will shall describe explain tell give list show summarize compare""".split())


class DomainModel:
    """Corpus-vs-general-English word likelihood ratio."""

    def __init__(self, corpus: Corpus, extra_stop: frozenset[str]) -> None:
        self.counts = corpus.token_counts
        self.total = sum(self.counts.values())
        self.stop = STOPWORDS | _QUESTION_WORDS | extra_stop

    def word_llr(self, w: str) -> float | None:
        c = self.counts.get(w, 0)
        g = word_frequency(w, "en") if word_frequency else 0.0
        if c == 0 and g == 0:
            return None  # typo / rare name: no evidence either way
        if word_frequency is None:
            return 2.0 if c else -2.0
        pc = (c + 0.5) / (self.total + 1)
        pg = max(g, 1e-8)
        return math.log(pc / pg)

    def score(self, text: str) -> tuple[float, list[tuple[str, float]]]:
        words = [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 2 and w not in self.stop]
        scored = [(w, llr) for w in dict.fromkeys(words) if (llr := self.word_llr(w)) is not None]
        if not scored:
            return 0.5, []
        vals = [1.0 / (1.0 + math.exp(-llr)) for _, llr in scored]
        return float(np.mean(vals)), [(w, round(llr, 2)) for w, llr in scored]


class ScopeGuard(BaseGuard):
    name = "scope"
    stage = "input"
    fail_action = "block"

    def __init__(self, config: dict | None, corpus: Corpus, embedder: Embedder, registry: CompanyRegistry) -> None:
        super().__init__(config)
        self.corpus, self.embedder, self.registry = corpus, embedder, registry
        self.threshold = float(self.config.get("threshold", 0.35))
        w = self.config.get("weights", {}) or {}
        self.w_dom = float(w.get("domain", 0.55))
        self.w_sim = float(w.get("similarity", 0.30))
        self.w_co = float(w.get("company", 0.15))
        self.sim_floor = float(self.config.get("similarity_floor", 0.10))
        self.sim_ceil = float(self.config.get("similarity_ceiling", 0.50))
        self.domain = DomainModel(corpus, registry.all_alias_tokens())

    @cached_property
    def centroids(self) -> tuple[np.ndarray, list[tuple[str, str]]]:
        mat = self.embedder.chunk_matrix
        keys: dict[tuple[str, str], list[int]] = {}
        for c in self.corpus.chunks:
            keys.setdefault((c.ticker, c.section), []).append(c.index)
        labels, rows = [], []
        for k, idx in keys.items():
            if len(idx) < 2:
                continue
            v = mat[idx].mean(axis=0)
            rows.append(v / (np.linalg.norm(v) + 1e-9))
            labels.append(k)
        return np.vstack(rows).astype(np.float32), labels

    def query_vec(self, ctx: GuardContext) -> np.ndarray:
        key = ("qvec", ctx.effective_question)
        if key not in ctx.cache:
            ctx.cache[key] = self.embedder.encode([ctx.effective_question])[0]
        return ctx.cache[key]

    def evaluate(self, ctx: GuardContext) -> tuple[bool, float | None, dict[str, Any]]:
        text = ctx.question
        mentions = self.registry.detect(text)
        dom, words = self.domain.score(text)
        cents, labels = self.centroids
        sims = cents @ self.query_vec(ctx)
        best = int(np.argmax(sims))
        sim = float(sims[best])
        sim_norm = float(np.clip((sim - self.sim_floor) / max(self.sim_ceil - self.sim_floor, 1e-6), 0.0, 1.0))
        score = self.w_dom * dom + self.w_sim * sim_norm + self.w_co * (1.0 if mentions.covered else 0.0)
        reason = None
        if mentions.unknown_only:
            reason = "unknown_company"
        elif score < self.threshold:
            reason = "off_topic"
        detail = {
            "reason": reason,
            "domain": round(dom, 4),
            "similarity": round(sim, 4),
            "similarity_norm": round(sim_norm, 4),
            "nearest": f"{labels[best][0]}/{labels[best][1]}",
            "covered_companies": list(mentions.covered),
            "other_companies": list(mentions.other),
            "words": words[:12],
        }
        return reason is None, float(score), detail
