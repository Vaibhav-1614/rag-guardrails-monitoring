"""Hybrid retrieval over the 10-K corpus: BM25 + dense, fused with weighted RRF.

Ranking uses RRF (rank-based, so its values carry no absolute meaning). The
retrieval-confidence gate needs an *absolute* relevance score, so every returned chunk
also gets ``relevance = w_lex * lexical + w_sem * semantic`` where

- lexical  = IDF-weighted share of the question's content terms that occur in the chunk
- semantic = cosine(query, chunk) under the active embedder (clipped at 0)

Both are in [0, 1] and comparable across queries.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from rank_bm25 import BM25Okapi

from rag.companies import CompanyRegistry, get_registry
from rag.config import load_settings
from rag.corpus import Corpus, load_corpus
from rag.embedder import Embedder, get_embedder
from rag.text import FILING_BOILERPLATE, content_terms, stem


@dataclass
class RetrievedChunk:
    chunk_id: str
    ticker: str
    section: str
    text: str
    rank: int
    bm25: float
    dense: float
    lexical: float
    relevance: float
    synthetic: bool = False  # planted by a test harness (indirect-injection eval)

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    tickers: list[str]
    strategy: str
    embedder: str
    latency_ms: float
    query_terms: list[str] = field(default_factory=list)

    @property
    def top_score(self) -> float:
        return max((c.relevance for c in self.chunks), default=0.0)

    def n_above(self, floor: float) -> int:
        return sum(1 for c in self.chunks if c.relevance >= floor)


class Retriever:
    def __init__(
        self,
        corpus: Corpus | None = None,
        embedder: Embedder | None = None,
        registry: CompanyRegistry | None = None,
        settings: dict | None = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.corpus = corpus or load_corpus(self.settings)
        self.embedder = embedder or get_embedder(self.settings)
        self.registry = registry or get_registry(self.settings)
        r = self.settings.get("retrieval", {})
        self.strategy = r.get("strategy", "hybrid")
        self.top_k = int(r.get("top_k", 5))
        self.candidate_k = int(r.get("candidate_k", 20))
        self.rrf_k = int(r.get("rrf_k", 60))
        w = r.get("rrf_weights", {}) or {}
        self.w_bm25, self.w_dense = float(w.get("bm25", 1.0)), float(w.get("dense", 0.5))
        cw = r.get("confidence_weights", {}) or {}
        self.w_lex, self.w_sem = float(cw.get("lexical", 0.6)), float(cw.get("semantic", 0.4))
        self.bm25 = BM25Okapi(self.corpus.tokenized)
        # Stemmed term sets for the confidence score ("spend"/"spent", "repurchase"/"repurchases").
        self._doc_sets = [{stem(t) for t in toks} for toks in self.corpus.tokenized]
        self._stop = FILING_BOILERPLATE | self.registry.all_alias_tokens()

    @cached_property
    def idf(self) -> dict[str, float]:
        """IDF per stemmed term (max over surface forms), floored so very common terms keep some weight."""
        floor = 0.05
        out: dict[str, float] = {}
        for t, v in self.bm25.idf.items():
            s = stem(t)
            out[s] = max(out.get(s, floor), v, floor)
        return out

    def query_terms(self, question: str) -> list[str]:
        return list(dict.fromkeys(stem(t) for t in content_terms(question, self._stop)))

    def lexical_coverage(self, terms: list[str], doc_index: int) -> float:
        if not terms:
            return 0.0
        doc = self._doc_sets[doc_index]
        # Unknown terms get the max IDF: a word the corpus has never seen is maximally informative.
        max_idf = max(self.idf.values()) if self.idf else 1.0
        weights = {t: self.idf.get(t, max_idf) for t in terms}
        total = sum(weights.values())
        return sum(w for t, w in weights.items() if t in doc) / total if total else 0.0

    def retrieve(
        self,
        question: str,
        tickers: list[str] | tuple[str, ...] | None = None,
        top_k: int | None = None,
        query_vec: np.ndarray | None = None,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        k = top_k or self.top_k
        if tickers is None:
            tickers = list(self.registry.detect(question).covered)
        tickers = [t for t in tickers if t in self.registry.covered]
        mask = self.corpus.mask(tickers)
        terms = self.query_terms(question)

        qv = query_vec if query_vec is not None else self.embedder.encode([question])[0]
        dense = self.embedder.chunk_matrix @ qv
        bm = np.asarray(self.bm25.get_scores(content_terms(question, self._stop) or question.lower().split()))
        dense_m = np.where(mask, dense, -np.inf)
        # Chunks sharing no term with the query have BM25 = 0; they must not enter the BM25 list
        # (otherwise zero-score ties pad it and earn RRF credit).
        bm_m = np.where(mask & (bm > 0), bm, -np.inf)

        if self.strategy == "bm25":
            order = self._top(bm_m, len(self.corpus))
            fused = {int(i): 1.0 / (self.rrf_k + r) for r, i in enumerate(order, 1)}
        elif self.strategy == "dense":
            order = self._top(dense_m, len(self.corpus))
            fused = {int(i): 1.0 / (self.rrf_k + r) for r, i in enumerate(order, 1)}
        else:
            fused = {}
            for weight, scores in ((self.w_bm25, bm_m), (self.w_dense, dense_m)):
                for r, i in enumerate(self._top(scores, self.candidate_k), 1):
                    fused[int(i)] = fused.get(int(i), 0.0) + weight / (self.rrf_k + r)
        ranked = sorted(fused, key=lambda i: -fused[i])
        ranked = [i for i in ranked if mask[i]]
        chosen = self._diversify(ranked, tickers, k)

        out: list[RetrievedChunk] = []
        for rank, i in enumerate(chosen, 1):
            c = self.corpus.chunks[i]
            lex = self.lexical_coverage(terms, i)
            sem = max(float(dense[i]), 0.0)
            out.append(
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    ticker=c.ticker,
                    section=c.section,
                    text=c.text,
                    rank=rank,
                    bm25=float(bm[i]),
                    dense=float(dense[i]),
                    lexical=lex,
                    relevance=self.w_lex * lex + self.w_sem * sem,
                )
            )
        return RetrievalResult(
            chunks=out,
            tickers=list(tickers),
            strategy=self.strategy,
            embedder=self.embedder.name,
            latency_ms=(time.perf_counter() - t0) * 1000,
            query_terms=terms,
        )

    @staticmethod
    def _top(scores: np.ndarray, n: int) -> np.ndarray:
        finite = np.isfinite(scores)
        n = min(n, int(finite.sum()))
        if n <= 0:
            return np.array([], dtype=int)
        idx = np.argpartition(-np.where(finite, scores, -1e18), n - 1)[:n]
        return idx[np.argsort(-scores[idx], kind="stable")]

    def _diversify(self, ranked: list[int], tickers: list[str], k: int) -> list[int]:
        """For multi-company questions, give each named company a fair share of the top-k."""
        if len(tickers) <= 1:
            return ranked[:k]
        quota = max(1, k // len(tickers))
        picked: list[int] = []
        for t in tickers:
            picked += [i for i in ranked if self.corpus.chunks[i].ticker == t][:quota]
        for i in ranked:
            if len(picked) >= k:
                break
            if i not in picked:
                picked.append(i)
        return sorted(picked[:k], key=ranked.index)
