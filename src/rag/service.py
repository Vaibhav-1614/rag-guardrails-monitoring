"""RAG service orchestrator: input guards -> retrieval -> confidence gate -> sanitizer -> LLM ->
output guards -> (regenerate once) -> response, with one trace per request."""
from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from guards.base import GuardContext, GuardResult
from guards.pipeline import GuardPipeline
from rag import prompts
from rag.companies import CompanyRegistry, get_registry
from rag.config import load_settings
from rag.corpus import Corpus, load_corpus
from rag.embedder import Embedder, get_embedder
from rag.llm import GenerationRequest, Generator, get_generators
from rag.retriever import RetrievedChunk, Retriever
from rag.tracing import Tracer, TraceRecord, make_backend

logger = logging.getLogger(__name__)


class Citation(BaseModel):
    n: int
    chunk_id: str
    ticker: str
    section: str


class ChunkOut(BaseModel):
    n: int
    chunk_id: str
    ticker: str
    section: str
    relevance: float
    text: str


class AskResponse(BaseModel):
    trace_id: str
    question: str
    answer: str
    status: str                      # answered | refused | fallback | error
    refusal_reason: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    chunks: list[ChunkOut] = Field(default_factory=list)
    guards: list[GuardResult] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    regenerated: bool = False
    provider: str | None = None
    model: str | None = None
    latency_ms: dict[str, float] = Field(default_factory=dict)
    config_version: str = ""
    _trace: TraceRecord | None = PrivateAttr(default=None)

    @property
    def trace_record(self) -> TraceRecord | None:
        return self._trace


@dataclass
class ServiceComponents:
    settings: dict
    corpus: Corpus
    embedder: Embedder
    registry: CompanyRegistry
    retriever: Retriever
    pipeline: GuardPipeline
    generator: Generator
    alternate: Generator
    tracer: Tracer
    extras: dict[str, Any] = field(default_factory=dict)


class RAGService:
    def __init__(self, components: ServiceComponents) -> None:
        self.c = components
        self.companies = [f"{components.registry.name(t)} ({t})" for t in components.registry.tickers]
        self.system_prompt = prompts.build_system_prompt(self.companies)

    @classmethod
    def build(cls, settings: dict | None = None, guards_config: str | None = None, generator: Generator | None = None,
              tracer: Tracer | None = None, trace_backend: str | None = None,
              guard_overrides: dict[str, Any] | None = None) -> RAGService:
        settings = settings or load_settings()
        corpus = load_corpus(settings)
        embedder = get_embedder(settings)
        registry = get_registry(settings)
        retriever = Retriever(corpus, embedder, registry, settings)
        pipeline = GuardPipeline.from_file(
            guards_config or settings.get("guards_config", "configs/guards/v2.yaml"), corpus, embedder, registry,
            allowlist_path=settings.get("pii_allowlist", "configs/pii_allowlist.txt"), overrides=guard_overrides,
        )
        primary, alternate = get_generators(settings)
        if generator is not None:
            primary = alternate = generator
        if tracer is None:
            env = settings.get("tracing", {}).get("environment", "production")
            tracer = Tracer(make_backend(settings, trace_backend), mask=pipeline.mask, environment=env)
        return cls(ServiceComponents(settings, corpus, embedder, registry, retriever, pipeline, primary, alternate, tracer))

    # ------------------------------------------------------------------ main entry point
    def ask(self, question: str, tickers: list[str] | None = None, *, environment: str | None = None,
            timestamp: datetime | None = None, extra_chunks: list[dict] | None = None,
            replace_chunks: list[dict] | None = None, trace: bool | str = True, tags: list[str] | None = None) -> AskResponse:
        """Answer ``question``.

        ``trace``: True writes the trace now; "defer" only builds it (``resp.trace_record``) so the caller
        can write it later (the API does this in a background task); False skips tracing.
        ``extra_chunks`` / ``replace_chunks`` let the indirect-injection harness plant synthetic chunks.
        """
        t0 = time.perf_counter()
        lat: dict[str, float] = {}
        trace_id = uuid.uuid4().hex
        ctx = GuardContext(question=question, trace_id=trace_id)
        p = self.c.pipeline
        guard_results: list[GuardResult] = []
        resp = AskResponse(trace_id=trace_id, question=question, answer="", status="answered", config_version=p.version)

        # 1. input guards (parallel)
        t = time.perf_counter()
        inp = p.run_input(ctx)
        lat["input_guards"] = (time.perf_counter() - t) * 1000
        guard_results += inp.results
        if inp.blocked:
            resp.answer, resp.status, resp.refusal_reason = inp.refusal or prompts.POLICY_REFUSAL, "refused", inp.blocked_by
            return self._finish(resp, ctx, guard_results, None, lat, t0, environment, timestamp, trace, tags)

        # 2. retrieval (reuses the query vector the scope guard computed)
        query = ctx.effective_question
        search_query = search_text(query)
        qvec = ctx.cache.get(("qvec", query)) if search_query == query else None
        retrieval = self.c.retriever.retrieve(search_query, tickers or ctx.tickers or None, query_vec=qvec)
        lat["retrieval"] = retrieval.latency_ms
        if replace_chunks is not None:
            retrieval.chunks = [self._synthetic(c, i) for i, c in enumerate(replace_chunks, 1)]
        elif extra_chunks:
            planted = [self._synthetic(c, i) for i, c in enumerate(extra_chunks, 1)]
            retrieval.chunks = (planted + retrieval.chunks)[: max(len(retrieval.chunks), len(planted))]
        ctx.retrieval = retrieval
        ctx.chunks = [{"chunk_id": c.chunk_id, "ticker": c.ticker, "section": c.section, "text": c.text} for c in retrieval.chunks]

        # 3. retrieval-confidence gate
        rc = p.run_retrieval(ctx)
        if rc is not None:
            guard_results.append(rc)
            if rc.enforced:
                resp.answer, resp.status, resp.refusal_reason = prompts.NOT_FOUND, "refused", "retrieval_conf"
                return self._finish(resp, ctx, guard_results, retrieval, lat, t0, environment, timestamp, trace, tags)

        # 4. context sanitizer
        san = p.run_context(ctx)
        if san is not None:
            guard_results.append(san)

        # 5. generation + output guards, regenerating once on enforced failures
        regen_with = p.cfg("citation").get("regenerate_with", "same")
        reasons: list[str] = []
        for attempt in (0, 1):
            ctx.attempt = attempt
            gen = self.c.generator if attempt == 0 or regen_with == "same" else self.c.alternate
            user = prompts.build_user_prompt(query, ctx.chunks)
            system = self.system_prompt + (prompts.strict_suffix(reasons) if attempt else "")
            t = time.perf_counter()
            result = gen.generate(GenerationRequest(query, ctx.chunks, system, user, strict=bool(attempt)))
            lat[f"generation_{attempt}"] = (time.perf_counter() - t) * 1000
            resp.provider, resp.model = result.provider, result.model
            resp.latency_ms.setdefault("tokens_in", 0)
            resp.latency_ms["tokens_in"] += result.input_tokens
            resp.latency_ms.setdefault("tokens_out", 0)
            resp.latency_ms["tokens_out"] += result.output_tokens
            if result.error and not result.text:
                resp.answer, resp.status, resp.refusal_reason = prompts.SAFE_FALLBACK, "error", "generation_error"
                resp.flags.append(f"error:{result.error[:120]}")
                return self._finish(resp, ctx, guard_results, retrieval, lat, t0, environment, timestamp, trace, tags)
            ctx.answer = result.text
            t = time.perf_counter()
            out = p.run_output(ctx)
            lat[f"output_guards_{attempt}"] = (time.perf_counter() - t) * 1000
            for r in out.results:
                r.detail["attempt"] = attempt
            guard_results += out.results
            if out.decision == "regenerate":
                reasons = out.reasons
                resp.regenerated = True
                continue
            resp.answer = out.answer or ctx.answer
            resp.flags += out.flags
            if out.fallback in ("safe_fallback", "advice_refusal"):
                resp.status = "fallback" if out.fallback == "safe_fallback" else "refused"
                resp.refusal_reason = out.fallback
            elif out.fallback == "strip_uncited":
                resp.flags.append("stripped_uncited_sentences")
            break
        if resp.answer.strip() == prompts.NOT_FOUND:
            resp.status, resp.refusal_reason = "refused", "model_not_found"
        return self._finish(resp, ctx, guard_results, retrieval, lat, t0, environment, timestamp, trace, tags)

    @staticmethod
    def _synthetic(c: dict, i: int) -> RetrievedChunk:
        return RetrievedChunk(chunk_id=c.get("chunk_id", f"SYNTH_{i}"), ticker=c.get("ticker", "AAPL"),
                              section=c.get("section", "Item 7"), text=c["text"], rank=i, bm25=0.0, dense=0.0,
                              lexical=1.0, relevance=1.0, synthetic=True)

    # ------------------------------------------------------------------ response + trace
    def _finish(self, resp: AskResponse, ctx: GuardContext, results: list[GuardResult], retrieval, lat: dict,
                t0: float, environment: str | None, timestamp: datetime | None, trace: bool | str,
                tags: list[str] | None) -> AskResponse:
        from guards.citation import cited_ids

        lat["total"] = (time.perf_counter() - t0) * 1000
        lat["guards_total"] = sum(r.latency_ms for r in results)
        resp.latency_ms.update({k: round(v, 2) for k, v in lat.items()})
        resp.guards = results
        if retrieval is not None:
            resp.chunks = [ChunkOut(n=i, chunk_id=c.chunk_id, ticker=c.ticker, section=c.section,
                                    relevance=round(c.relevance, 4), text=ctx.chunks[i - 1]["text"] if i <= len(ctx.chunks) else c.text)
                           for i, c in enumerate(retrieval.chunks, 1)]
            used = sorted({i for i in cited_ids(resp.answer) if 1 <= i <= len(resp.chunks)})
            resp.citations = [Citation(n=i, chunk_id=resp.chunks[i - 1].chunk_id, ticker=resp.chunks[i - 1].ticker,
                                       section=resp.chunks[i - 1].section) for i in used]
        if trace:
            resp._trace = self.trace_record(resp, ctx, retrieval, environment, timestamp, tags)
            if trace is True:
                self.c.tracer.log(resp._trace)
        return resp

    def trace_record(self, resp: AskResponse, ctx: GuardContext, retrieval, environment: str | None,
                     timestamp: datetime | None, tags: list[str] | None) -> TraceRecord:
        by_name: dict[str, GuardResult] = {}
        for r in resp.guards:  # keep the final attempt's result per guard
            by_name[r.guard] = r
        final = {n: r.compact() for n, r in by_name.items()}
        num = by_name.get("numeric_grounding")
        nli = by_name.get("nli_grounding")
        cit = by_name.get("citation")
        scope = by_name.get("scope")
        inj = by_name.get("prompt_injection")
        metadata: dict[str, Any] = {
            "status": resp.status,
            "refusal_reason": resp.refusal_reason,
            "config_version": resp.config_version,
            "guards": final,
            "guard_trigger": sorted(n for n, r in by_name.items() if not r.passed),
            "guard_enforced": sorted(n for n, r in by_name.items() if r.enforced),
            "regenerated": resp.regenerated,
            "flags": resp.flags,
            "tickers": ctx.tickers,
            "provider": resp.provider,
            "model": resp.model,
            "embedder": self.c.embedder.name,
            "latency_ms": resp.latency_ms,
            "answer_chars": len(resp.answer),
            "n_citations": len(cited_ids_safe(resp.answer)),
            "unsupported_numbers": len(num.detail.get("unsupported", [])) if num else None,
            "n_numbers": num.detail.get("n_numbers") if num else None,
            "pii_entities": [e["entity_type"] for e in (by_name["pii"].detail.get("entities", []) if "pii" in by_name else [])],
            "scope_reason": scope.detail.get("reason") if scope else None,
            "injection_rules": [r["rule"] for r in inj.detail.get("rules", [])][:5] if inj else [],
        }
        if retrieval is not None:
            metadata["retrieval"] = {"top_score": round(retrieval.top_score, 4), "embedder": retrieval.embedder,
                                     "chunk_ids": [c.chunk_id for c in retrieval.chunks],
                                     "scores": [round(c.relevance, 4) for c in retrieval.chunks]}
        scores: dict[str, float] = {}
        if scope is not None and scope.score is not None:
            scores["scope"] = float(scope.score)
        if retrieval is not None:
            scores["retrieval_top"] = float(retrieval.top_score)
        if nli is not None and nli.score is not None:
            scores["groundedness"] = float(nli.score)
        if num is not None and num.score is not None:
            scores["numeric_support"] = float(num.score)
        if cit is not None and cit.score is not None:
            scores["citation_coverage"] = float(cit.score)
        if inj is not None and inj.score is not None:
            scores["injection"] = float(inj.score)
        rec = TraceRecord(
            id=resp.trace_id,
            environment=environment or self.c.tracer.environment,
            input={"question": ctx.effective_question},
            output={"answer": resp.answer},
            metadata=metadata,
            scores=scores,
            tags=list(tags or []) + [f"status:{resp.status}"],
        )
        if timestamp is not None:
            rec.timestamp = timestamp
        return rec


_PLACEHOLDER = re.compile(r"<[A-Z_]+>")


def search_text(question: str) -> str:
    """Retrieval query without the clauses that carried redacted PII ("My email is <EMAIL>, ..."):
    their words ("email", "phone") would only drag retrieval confidence down."""
    if not _PLACEHOLDER.search(question):
        return question
    clauses = re.split(r"(?<=[,;.!?:])\s+|\s+-\s+", question)
    kept = [c for c in clauses if not _PLACEHOLDER.search(c)]
    return " ".join(kept) if kept else _PLACEHOLDER.sub(" ", question)


def cited_ids_safe(text: str) -> list[int]:
    from guards.citation import cited_ids

    try:
        return cited_ids(text)
    except ValueError:
        return []
