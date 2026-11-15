"""Answer generators.

- GeminiGenerator / GroqGenerator: free-tier LLM APIs (keys from GOOGLE_API_KEY / GROQ_API_KEY).
- ExtractiveGenerator: offline, deterministic. Picks the retrieved sentences that best cover the
  question and cites them. Used when no key is set (CI, local dev) and as the last-resort fallback.
- FaultInjectingGenerator: wraps another generator and corrupts numbers / drops citations at a set
  rate. Only for tests and the traffic simulator's "upstream model drift" scenario.
"""
from __future__ import annotations

import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from rag.config import load_settings
from rag.prompts import NOT_FOUND

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    question: str
    chunks: list[dict]
    system: str
    user: str
    strict: bool = False
    meta: dict = field(default_factory=dict)


@dataclass
class GenerationResult:
    text: str
    provider: str
    model: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


class Generator(Protocol):
    provider: str
    model: str

    def generate(self, req: GenerationRequest) -> GenerationResult: ...


def _post_with_retry(url: str, *, json: dict, headers: dict, timeout: float, retries: int = 2) -> httpx.Response:
    delay = 2.0
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = httpx.post(url, json=json, headers=headers, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            r.raise_for_status()
            return r
        except httpx.HTTPError as e:
            last = e
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
    raise last or RuntimeError("request failed")


class GeminiGenerator:
    provider = "gemini"

    def __init__(self, model: str, api_key: str, temperature: float = 0.0, max_tokens: int = 600, timeout: float = 60) -> None:
        self.model, self._key, self.temperature, self.max_tokens, self.timeout = model, api_key, temperature, max_tokens, timeout

    def generate(self, req: GenerationRequest) -> GenerationResult:
        t0 = time.perf_counter()
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": req.system}]},
            "contents": [{"role": "user", "parts": [{"text": req.user}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        try:
            r = _post_with_retry(url, json=body, headers={"x-goog-api-key": self._key}, timeout=self.timeout)
            data = r.json()
            parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts).strip()
            usage = data.get("usageMetadata", {})
            return GenerationResult(
                text, self.provider, self.model, (time.perf_counter() - t0) * 1000,
                int(usage.get("promptTokenCount", 0)), int(usage.get("candidatesTokenCount", 0)),
            )
        except Exception as e:
            return GenerationResult("", self.provider, self.model, (time.perf_counter() - t0) * 1000, error=f"{type(e).__name__}: {e}")


class GroqGenerator:
    provider = "groq"

    def __init__(self, model: str, api_key: str, temperature: float = 0.0, max_tokens: int = 600, timeout: float = 60) -> None:
        self.model, self._key, self.temperature, self.max_tokens, self.timeout = model, api_key, temperature, max_tokens, timeout

    def generate(self, req: GenerationRequest) -> GenerationResult:
        t0 = time.perf_counter()
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": req.system}, {"role": "user", "content": req.user}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        try:
            r = _post_with_retry(
                "https://api.groq.com/openai/v1/chat/completions",
                json=body, headers={"Authorization": f"Bearer {self._key}"}, timeout=self.timeout,
            )
            data = r.json()
            text = (data["choices"][0]["message"]["content"] or "").strip()
            usage = data.get("usage", {})
            return GenerationResult(
                text, self.provider, self.model, (time.perf_counter() - t0) * 1000,
                int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)),
            )
        except Exception as e:
            return GenerationResult("", self.provider, self.model, (time.perf_counter() - t0) * 1000, error=f"{type(e).__name__}: {e}")


class ExtractiveGenerator:
    """Deterministic no-LLM answerer, the same path the upstream service uses without API keys:
    the facts-table sentence when the question is about a company-wide figure, otherwise the
    retrieved sentences (or table rows with their header) that best match the question, each cited."""

    provider = "extractive"
    model = "extractive-v2"

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def generate(self, req: GenerationRequest) -> GenerationResult:
        from rag.upstream import ensure_importable

        ensure_importable()
        from rag_service.generation import extractive_answer
        from rag_service.retriever import Passage

        t0 = time.perf_counter()
        text = req.meta.get("direct_answer") or ""
        if not text:
            passages = [Passage(chunk_id=c.get("chunk_id", f"c{i}"), ticker=c.get("ticker", ""), text=c["text"], score=0.0,
                                rank=i, kind=c.get("kind", "text")) for i, c in enumerate(req.chunks, 1)]
            text, _ = extractive_answer(req.question, passages, self.max_sentences, scorer=req.meta.get("scorer"))
        return GenerationResult(cite_every_sentence(text) or NOT_FOUND, self.provider, self.model, (time.perf_counter() - t0) * 1000)


_UNIT = re.compile(r"(.+?)((?:\s*\[\d+\])+)\.?(?=\s|$)", re.S)


def cite_every_sentence(text: str) -> str:
    """Upstream cites once per quoted unit (which can hold two sentences) and leaves the closing
    sentence of a comparison ("Apple's was higher.") uncited. Give every sentence its marker so the
    answer meets the per-sentence citation rule the guards enforce."""
    from rag.text import split_sentences

    out: list[str] = []
    last_end, all_ids = 0, []
    for m in _UNIT.finditer(text):
        marker = "".join(m.group(2).split())
        all_ids += [x for x in re.findall(r"\[\d+\]", marker) if x not in all_ids]
        for sent in split_sentences(m.group(1)) or [m.group(1).strip()]:
            out.append(f"{sent.strip().removesuffix('.')} {marker}.")
        last_end = m.end()
    tail = text[last_end:].strip()
    if tail and all_ids:  # derived conclusion: rests on every source cited before it
        out.append(f"{tail.removesuffix('.')} {''.join(all_ids)}.")
    elif tail:
        out.append(tail)
    return " ".join(out)


_NUM = re.compile(r"(?<![\w\[])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d{2,})(?![\w\]])")


class FaultInjectingGenerator:
    """Simulates a degraded upstream model: perturbs numbers and drops citations."""

    def __init__(self, base: Generator, number_rate: float = 0.5, citation_drop_rate: float = 0.5, seed: int = 0) -> None:
        self.base, self.number_rate, self.citation_drop_rate = base, number_rate, citation_drop_rate
        self.rng = random.Random(seed)
        self.provider, self.model = base.provider, f"{base.model}+faults"

    def _perturb(self, m: re.Match) -> str:
        s = m.group(1)
        if self.rng.random() >= self.number_rate or (len(s) == 4 and s.isdigit() and 1990 <= int(s) <= 2035):
            return s
        digits = [i for i, ch in enumerate(s) if ch.isdigit()]
        i = digits[min(len(digits) - 1, max(0, len(digits) // 2))]
        new = str((int(s[i]) + self.rng.choice([1, 2, 3, 4, 5])) % 10)
        if i == digits[0] and new == "0":
            new = "9"
        return s[:i] + new + s[i + 1 :]

    def generate(self, req: GenerationRequest) -> GenerationResult:
        res = self.base.generate(req)
        if res.error or res.text == NOT_FOUND:
            return res
        text = _NUM.sub(self._perturb, res.text)
        if self.rng.random() < self.citation_drop_rate:
            text = re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", text)
        res.text, res.provider, res.model = text, self.provider, self.model
        return res


class FallbackGenerator:
    """Use ``primary``; on an error (quota, outage) answer with ``fallback`` instead."""

    def __init__(self, primary: Generator, fallback: Generator | None) -> None:
        self.primary, self.fallback = primary, fallback
        self.provider, self.model = primary.provider, primary.model

    def generate(self, req: GenerationRequest) -> GenerationResult:
        res = self.primary.generate(req)
        if res.error and self.fallback is not None:
            logger.warning("primary generator failed (%s); using %s", res.error, self.fallback.provider)
            fb = self.fallback.generate(req)
            fb.error = None if not fb.error else f"primary: {res.error}; fallback: {fb.error}"
            return fb
        return res


def make_generator(provider: str, settings: dict | None = None) -> Generator:
    settings = settings or load_settings()
    g = settings.get("generation", {})
    kw = dict(
        temperature=float(g.get("temperature", 0.0)),
        max_tokens=int(g.get("max_output_tokens", 600)),
        timeout=float(g.get("timeout_s", 60)),
    )
    if provider == "gemini":
        key = _gemini_key()
        if not key:
            raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY is not set")
        return GeminiGenerator(g.get("gemini_model", "gemini-2.5-flash"), key, **kw)
    if provider == "groq":
        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set")
        return GroqGenerator(g.get("groq_model", "openai/gpt-oss-120b"), key, **kw)
    if provider == "extractive":
        return ExtractiveGenerator()
    raise ValueError(f"unknown provider {provider!r}")


def _gemini_key() -> str:
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()


def available_providers() -> list[str]:
    out = []
    # Same order as the upstream service: Groq first, then Gemini.
    if os.environ.get("GROQ_API_KEY", "").strip():
        out.append("groq")
    if _gemini_key():
        out.append("gemini")
    return out + ["extractive"]


def get_generators(settings: dict | None = None) -> tuple[Generator, Generator]:
    """(primary-with-fallback, alternate). The alternate is used for 'regenerate_with: fallback'."""
    settings = settings or load_settings()
    g = settings.get("generation", {})
    avail = available_providers()
    primary_name = g.get("provider", "auto")
    if primary_name == "auto":
        primary_name = avail[0]
    fb_name = g.get("fallback_provider", "auto")
    if fb_name == "auto":
        fb_name = next((p for p in avail if p != primary_name), "extractive")
    primary = make_generator(primary_name, settings)
    alternate = make_generator(fb_name, settings) if fb_name != primary_name else primary
    last_resort = ExtractiveGenerator() if "extractive" not in (primary_name, fb_name) else None
    fallback: Generator | None = alternate if alternate is not primary else None
    if last_resort is not None:
        fallback = FallbackGenerator(fallback, last_resort) if fallback else last_resort
    return FallbackGenerator(primary, fallback), alternate
