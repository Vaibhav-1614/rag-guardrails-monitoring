"""Guard protocol, result and context types shared by every guard."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

Action = Literal["allow", "redact", "block", "regenerate", "flag"]
Mode = Literal["monitor", "enforce", "off"]
Stage = Literal["input", "retrieval", "context", "output"]


class GuardResult(BaseModel):
    guard: str                      # "prompt_injection"
    stage: Stage = "input"
    passed: bool
    score: float | None = None      # classifier probability, similarity, fraction supported, ...
    action: Action = "allow"        # what the guard asks for; "allow" when it passed
    detail: dict[str, Any] = Field(default_factory=dict)  # matched spans, unsupported numbers, ...
    latency_ms: float = 0.0
    mode: Mode = "monitor"
    enforced: bool = False          # True when mode == enforce and the action was applied

    def compact(self) -> dict[str, Any]:
        """Small form stored in trace metadata."""
        d = {"guard": self.guard, "stage": self.stage, "passed": self.passed, "action": self.action,
             "mode": self.mode, "enforced": self.enforced, "latency_ms": round(self.latency_ms, 2)}
        if self.score is not None:
            d["score"] = round(float(self.score), 4)
        return d


@dataclass
class GuardContext:
    """Everything a guard may look at. Fields fill in as the request moves through the pipeline."""

    question: str
    redacted_question: str | None = None
    tickers: list[str] = field(default_factory=list)
    retrieval: Any = None                 # rag.retriever.RetrievalResult
    chunks: list[dict] = field(default_factory=list)  # chunks as sent to the LLM (after sanitizing)
    answer: str | None = None
    attempt: int = 0                      # 0 = first generation, 1 = regeneration
    trace_id: str | None = None
    cache: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_question(self) -> str:
        return self.redacted_question if self.redacted_question is not None else self.question


class Guard(Protocol):
    name: str
    stage: Stage

    def check(self, ctx: GuardContext) -> GuardResult: ...


class BaseGuard:
    """Convenience base: stores config and times ``evaluate``."""

    name: str = "base"
    stage: Stage = "input"
    fail_action: Action = "block"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.mode: Mode = self.config.get("mode", "monitor")
        self.fail_action = self.config.get("action", self.config.get("on_fail", self.fail_action))

    def evaluate(self, ctx: GuardContext) -> tuple[bool, float | None, dict[str, Any]]:
        raise NotImplementedError

    def check(self, ctx: GuardContext) -> GuardResult:
        t0 = time.perf_counter()
        passed, score, detail = self.evaluate(ctx)
        return GuardResult(
            guard=self.name,
            stage=self.stage,
            passed=passed,
            score=score,
            action="allow" if passed else self.fail_action,
            detail=detail,
            latency_ms=(time.perf_counter() - t0) * 1000,
            mode=self.mode,
            enforced=False,
        )
