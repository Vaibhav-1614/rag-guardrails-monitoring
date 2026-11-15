"""Retrieval adapter over the upstream (doc-01) pipeline.

Upstream does the work: company routing, the XBRL facts table for company-wide figures, stemmed
BM25 with query expansion over section/table-aware chunks, an optional cross-encoder re-ranker, and
the answerability decision (market data, forecasts, periods outside the filings, unknown entities,
low evidence). This module turns its result into what the guards need.

``confidence`` is the signal the retrieval-confidence gate and retrieval-drift monitoring use:
the re-ranker's evidence logit when the re-ranker is loaded, otherwise the share of the question's
terms found in the retrieved passages (upstream's ``coverage``). Facts-table hits count as 1.0.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from rag.config import load_settings
from rag.upstream import load_pipeline


@dataclass
class RetrievedChunk:
    chunk_id: str
    ticker: str
    section: str
    text: str
    rank: int
    relevance: float
    kind: str = "text"       # "fact" | "table" | "text"
    context: str = ""
    source_url: str | None = None
    synthetic: bool = False  # planted by a test harness (indirect-injection eval)

    def as_chunk(self) -> dict:
        return {"chunk_id": self.chunk_id, "ticker": self.ticker, "section": self.section, "text": self.text,
                "kind": self.kind, "context": self.context}


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    tickers: list[str]
    answerable: bool
    reason: str               # upstream's scope reason: ok | facts | market_data | forecast | low_evidence | ...
    confidence: float
    coverage: float
    evidence_score: float | None
    facts_mode: str           # single | ranking | none
    direct_answer: str | None  # the cited facts-table sentence for the no-LLM path
    version: str              # upstream retrieval config version (v2 | v2-bm25)
    latency_ms: float
    extra: dict = field(default_factory=dict)

    @property
    def top_score(self) -> float:
        return self.confidence


class Retriever:
    def __init__(self, settings: dict | None = None, pipeline=None) -> None:
        self.settings = settings or load_settings()
        self.pipeline = pipeline or load_pipeline(self.settings)
        self.version = self.pipeline.cfg.version
        self.reranker = bool(self.pipeline.cfg.rerank_model)
        # Cross-encoder sentence scorer for the no-LLM answer path (None without a re-ranker).
        self.scorer = getattr(self.pipeline.retriever, "rerank_scores", None) if self.reranker else None

    def retrieve(self, question: str, tickers: list[str] | None = None, top_k: int | None = None) -> RetrievalResult:
        t0 = time.perf_counter()
        prep = self.pipeline.prepare(question, tickers or None, top_k)
        chunks = [
            RetrievedChunk(chunk_id=p.chunk_id, ticker=p.ticker, section=p.section or "", text=p.text, rank=i,
                           relevance=float(p.score), kind=p.kind, context=p.context or "", source_url=p.source_url)
            for i, p in enumerate(prep.passages, 1)
        ]
        direct = self.pipeline._direct_answer(prep, self.scorer)
        if prep.hits:
            confidence = 1.0
        elif prep.evidence_score is not None:
            confidence = float(prep.evidence_score)
        else:
            confidence = float(prep.coverage)
        scope = [t for t in prep.scope if t != "ALL"]
        return RetrievalResult(
            chunks=chunks, tickers=scope, answerable=bool(prep.answerable), reason=prep.reason, confidence=confidence,
            coverage=float(prep.coverage), evidence_score=prep.evidence_score, facts_mode=prep.facts_mode,
            direct_answer=direct, version=self.version, latency_ms=(time.perf_counter() - t0) * 1000,
        )
