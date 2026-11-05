"""Guard pipeline: ordering, parallel input guards, modes, and the regenerate loop.

Modes (configs/guards/*.yaml):
- ``monitor``: the guard runs and its result is logged; the request is not affected (shadow mode).
- ``enforce``: the guard's action is applied.
- ``off``: the guard does not run.

Per-embedder threshold overrides live under ``by_embedder.<name>`` in a guard's config, so
thresholds calibrated for the LSA fallback and for bge can sit side by side.
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from guards.advice import AdviceGuard, AdviceOutputGuard
from guards.base import GuardContext, GuardResult
from guards.charset import LengthCharsetGuard
from guards.citation import CitationGuard, strip_uncited
from guards.injection import PromptInjectionGuard
from guards.nli import NLIGroundingGuard
from guards.numeric import NumericGroundingGuard
from guards.pii import PIIGuard, PIIOutputGuard, get_detector
from guards.retrieval_conf import RetrievalConfidenceGuard
from guards.sanitizer import ContextSanitizer
from guards.scope import ScopeGuard
from rag import prompts
from rag.companies import CompanyRegistry
from rag.config import load_yaml
from rag.corpus import Corpus
from rag.embedder import Embedder

INPUT_PRIORITY = ["charset", "prompt_injection", "advice", "scope"]


def resolve_config(raw: dict[str, Any], embedder_name: str) -> dict[str, Any]:
    cfg = copy.deepcopy(raw)
    for _name, g in (cfg.get("guards") or {}).items():
        overrides = (g.pop("by_embedder", None) or {}).get(embedder_name)
        if overrides:
            g.update(overrides)
    return cfg


@dataclass
class InputOutcome:
    results: list[GuardResult]
    blocked: bool = False
    blocked_by: str | None = None
    refusal: str | None = None


@dataclass
class OutputOutcome:
    results: list[GuardResult]
    decision: str = "allow"        # allow | regenerate | replace | flag
    answer: str | None = None
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    fallback: str | None = None    # which fallback was used, if any


class GuardPipeline:
    def __init__(self, config: dict[str, Any], corpus: Corpus, embedder: Embedder, registry: CompanyRegistry,
                 allowlist_path: str = "configs/pii_allowlist.txt", overrides: dict[str, Any] | None = None) -> None:
        self.raw_config = config
        self.config = resolve_config(config, embedder.name)
        g = self.config.get("guards", {})
        for name, ov in (overrides or {}).items():  # used by the eval harness
            g.setdefault(name, {}).update(ov)
        self.version = self.config.get("version", "unversioned")
        self.registry = registry
        self.corpus = corpus
        pii_cfg = {"allowlist": allowlist_path, **g.get("pii", {})}
        detector = get_detector(pii_cfg)
        self.pii_detector = detector
        out_pii_cfg = {**pii_cfg, **g.get("pii_output", {})}
        self.guards = {
            "charset": LengthCharsetGuard(g.get("charset", {"mode": "enforce"})),
            "pii": PIIGuard(pii_cfg, detector),
            "prompt_injection": PromptInjectionGuard(g.get("prompt_injection", {})),
            "scope": ScopeGuard(g.get("scope", {}), corpus, embedder, registry),
            "advice": AdviceGuard(g.get("advice", {})),
            "retrieval_conf": RetrievalConfidenceGuard(g.get("retrieval_conf", {})),
            "context_sanitizer": ContextSanitizer(g.get("context_sanitizer", {"mode": "enforce"})),
            "citation": CitationGuard({"mode": "enforce", "on_fail": "regenerate", "then": "strip_uncited", **g.get("citation", {})}),
            "numeric_grounding": NumericGroundingGuard(g.get("numeric_grounding", {}), corpus.table_scale, corpus.fiscal_years),
            "nli_grounding": NLIGroundingGuard(g.get("nli_grounding", {})),
            "advice_output": AdviceOutputGuard({**g.get("advice", {}), **g.get("advice_output", {})}),
            "pii_output": PIIOutputGuard(out_pii_cfg, detector),
        }
        self.input_names = ["charset", "pii", "prompt_injection", "scope", "advice"]
        self.output_names = ["citation", "numeric_grounding", "nli_grounding", "advice_output", "pii_output"]
        self._pool = ThreadPoolExecutor(max_workers=len(self.input_names), thread_name_prefix="input-guard")

    @classmethod
    def from_file(cls, path: str, corpus: Corpus, embedder: Embedder, registry: CompanyRegistry, **kw) -> GuardPipeline:
        return cls(load_yaml(path), corpus, embedder, registry, **kw)

    # ------------------------------------------------------------------ helpers
    def mode(self, name: str) -> str:
        return self.guards[name].mode

    def cfg(self, name: str) -> dict[str, Any]:
        return self.guards[name].config

    def modes(self) -> dict[str, str]:
        return {n: g.mode for n, g in self.guards.items()}

    def _run(self, name: str, ctx: GuardContext) -> GuardResult | None:
        guard = self.guards[name]
        if guard.mode == "off":
            return None
        res = guard.check(ctx)
        res.mode = guard.mode
        res.enforced = guard.mode == "enforce" and not res.passed and res.action != "allow"
        if name == "scope" and res.detail.get("reason") == "unknown_company":
            # Deterministic sub-check: can be enforced while the fuzzy score stays in shadow mode.
            sub = guard.config.get("unknown_company_mode", guard.mode)
            res.enforced = sub == "enforce"
            res.detail["unknown_company_mode"] = sub
        return res

    def mask(self, text: str) -> str:
        """PII redaction for logs/traces (always on, independent of guard modes)."""
        return self.pii_detector.redact(text)[0]

    # ------------------------------------------------------------------ stages
    def run_input(self, ctx: GuardContext) -> InputOutcome:
        futures = {n: self._pool.submit(self._run, n, ctx) for n in self.input_names}
        results = {n: f.result() for n, f in futures.items()}
        out = InputOutcome([r for r in results.values() if r is not None])
        pii = results.get("pii")
        if pii is not None and pii.enforced:
            ctx.redacted_question = pii.detail.get("redacted", ctx.question)
        for name in INPUT_PRIORITY:
            r = results.get(name)
            if r is not None and r.enforced and r.action == "block":
                out.blocked, out.blocked_by = True, name
                out.refusal = self._refusal(name, r)
                break
        ctx.tickers = list(self.registry.detect(ctx.question).covered)
        return out

    def _refusal(self, name: str, r: GuardResult) -> str:
        companies = [f"{self.registry.name(t)} ({t})" for t in self.registry.tickers]
        if name == "advice":
            return prompts.ADVICE_REFUSAL
        if name == "scope":
            if r.detail.get("reason") == "unknown_company":
                return prompts.unknown_company_message(r.detail.get("other_companies", []), self.registry.tickers)
            return prompts.out_of_scope_message(companies)
        return prompts.POLICY_REFUSAL

    def run_retrieval(self, ctx: GuardContext) -> GuardResult | None:
        return self._run("retrieval_conf", ctx)

    def run_context(self, ctx: GuardContext) -> GuardResult | None:
        res = self._run("context_sanitizer", ctx)
        if res is not None and res.mode == "enforce" and "sanitized_chunks" in ctx.cache:
            ctx.chunks = ctx.cache["sanitized_chunks"]
            res.enforced = not res.passed
        return res

    def run_output(self, ctx: GuardContext) -> OutputOutcome:
        results = {n: self._run(n, ctx) for n in self.output_names}
        out = OutputOutcome([r for r in results.values() if r is not None], answer=ctx.answer)
        advice = results.get("advice_output")
        if advice is not None and advice.enforced:
            out.decision, out.answer, out.fallback = "replace", prompts.ADVICE_REFUSAL, "advice_refusal"
            return self._final_pii(out, results)
        regen = [r for r in out.results if r.enforced and r.action == "regenerate"]
        if regen and ctx.attempt == 0:
            out.decision = "regenerate"
            out.reasons = [reason for r in regen for reason in r.detail.get("reasons", [f"{r.guard} failed"])]
            return out
        answer = ctx.answer or ""
        for r in regen:  # second failure: apply each guard's "then" policy
            then = self.cfg(r.guard).get("then", "fallback")
            if then == "flag":
                out.flags.append(f"unverified:{r.guard}")
                out.decision = "flag" if out.decision == "allow" else out.decision
            elif then == "strip_uncited" and not r.detail.get("invalid_citations"):
                stripped = strip_uncited(answer, self.cfg(r.guard).get("claims", "strict") == "strict")
                if stripped and stripped != answer:
                    answer, out.decision, out.fallback = stripped, "replace", "strip_uncited"
                elif not stripped:
                    answer, out.decision, out.fallback = prompts.SAFE_FALLBACK, "replace", "safe_fallback"
            else:
                answer, out.decision, out.fallback = prompts.SAFE_FALLBACK, "replace", "safe_fallback"
                break
        out.answer = answer
        for r in out.results:
            if r.enforced and r.action == "flag":
                out.flags.append(f"flag:{r.guard}")
                if out.decision == "allow":
                    out.decision = "flag"
        return self._final_pii(out, results)

    def _final_pii(self, out: OutputOutcome, results: dict[str, GuardResult | None]) -> OutputOutcome:
        pii = results.get("pii_output")
        if pii is not None and pii.enforced and out.answer:
            out.answer = self.pii_detector.redact(out.answer)[0]
            if out.decision == "allow":
                out.decision = "replace"
        return out

