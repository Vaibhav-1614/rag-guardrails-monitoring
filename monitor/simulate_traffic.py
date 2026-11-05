"""Traffic simulator (design doc §8.3): a portfolio service gets too little real traffic for drift
statistics to mean anything, so detectors are validated against labelled synthetic scenarios.

Scenarios (14 simulated days; days 0-6 are baseline traffic, the scenario starts on day 7):
- baseline:               paraphrases of golden questions + in-scope topic questions, every day
- topic_shift:            a growing share of questions about companies the corpus doesn't cover
- attack_wave:            a burst of prompt-injection attempts on the last day
- hallucination_pressure: a growing share of hallucination-bait questions
- model_drift:            the "provider" silently degrades the model from day 9 (numbers corrupted,
                          citations dropped), traffic unchanged

    python -m monitor.simulate_traffic --scenario topic_shift                     # in-process, local traces
    python -m monitor.simulate_traffic --scenario baseline --target https://... # deployed API (needs SIM_SECRET)

Traces are written to environment ``sim-<scenario>`` with backdated timestamps, never to production.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import httpx

from rag.config import resolve

COVERED = {"AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon", "META": "Meta", "NVDA": "NVIDIA",
           "TSLA": "Tesla", "JPM": "JPMorgan Chase", "JNJ": "Johnson & Johnson", "XOM": "Exxon Mobil"}
TOPICS = ["competition", "its main risk factors", "cybersecurity", "legal proceedings", "share repurchases",
          "research and development", "its employees", "supply chain risks", "climate-related risks", "capital expenditures",
          "revenue by segment", "foreign currency exposure", "investments in AI", "government regulation", "liquidity and debt",
          "dividends", "its business strategy", "tax matters", "goodwill and intangible assets", "critical accounting estimates"]
TOPIC_TEMPLATES = ["What does {c} say about {t}?", "How does {c} describe {t}?", "Summarize {c}'s disclosure on {t}.",
                   "What does the 10-K of {c} say about {t}?", "Tell me about {c}'s {t}."]
PREFIXES = ["", "", "", "Can you tell me: ", "Quick question - ", "According to the latest 10-K, ", "From the annual report, ",
            "I'd like to know: "]
OTHER = {
    "Netflix": ["paid memberships", "content spending", "subscriber growth in Asia"],
    "Coca-Cola": ["unit case volume", "bottling operations", "sugar taxes"],
    "Walmart": ["same-store sales", "e-commerce growth", "store count"],
    "Boeing": ["aircraft deliveries", "the 737 MAX program", "defense backlog"],
    "Pfizer": ["vaccine revenue", "drug pipeline", "patent cliffs"],
    "Disney": ["theme park attendance", "streaming losses", "box office results"],
    "Starbucks": ["store openings in China", "loyalty program members", "coffee prices"],
    "Uber": ["gross bookings", "driver classification", "delivery growth"],
    "Oracle": ["cloud infrastructure backlog", "database licenses", "data center leases"],
    "Intel": ["foundry losses", "chip manufacturing delays", "CHIPS Act grants"],
    "Nike": ["sneaker inventory", "wholesale partners", "running shoe sales"],
    "Salesforce": ["remaining performance obligations", "Slack revenue", "AI agents"],
}
OTHER_TEMPLATES = ["What does {o} report about {x}?", "How has {o}'s {x} changed?", "What did {o} say about {x} in its 10-K?",
                   "Give me {o}'s latest numbers on {x}."]
GENERAL = ["What are current mortgage rates?", "How do I open a Roth IRA?", "What is the price of Bitcoin today?",
           "Explain how index funds work.", "What is the Fed's next interest rate decision?", "How do I file my taxes online?",
           "Write a haiku about the stock market.", "Which credit card has the best travel rewards?"]
BAIT_YEARS = list(range(2008, 2020))


def _load(path: str) -> list[dict]:
    return [json.loads(line) for line in resolve(path).read_text().splitlines() if line.strip()]


@dataclass
class Pools:
    golden: list[str]
    attacks: list[str]
    bait: list[str]

    @classmethod
    def load(cls) -> Pools:
        golden = [r["question"] for r in _load("eval/golden.jsonl") if r["answerable"]]
        attacks = [r["text"] for r in _load("eval/redteam/injection.jsonl") if r["label"] == "attack"]
        bait = [r["question"] for r in _load("eval/redteam/hallucination_bait.jsonl")]
        return cls(golden, attacks, bait)


def _prefix(rng: random.Random, q: str) -> str:
    p = rng.choice(PREFIXES)
    return p + (q[0].lower() + q[1:] if p and not q[:2].isupper() else q)


def baseline_question(rng: random.Random, pools: Pools) -> str:
    if rng.random() < 0.5:
        q = rng.choice(pools.golden)
        if rng.random() < 0.2:
            for t, name in COVERED.items():
                if name in q:
                    q = q.replace(name, t, 1)
                    break
        return _prefix(rng, q)
    c = rng.choice(list(COVERED.values()))
    return _prefix(rng, rng.choice(TOPIC_TEMPLATES).format(c=c, t=rng.choice(TOPICS)))


def other_company_question(rng: random.Random) -> str:
    o = rng.choice(list(OTHER))
    return _prefix(rng, rng.choice(OTHER_TEMPLATES).format(o=o, x=rng.choice(OTHER[o])))


def bait_question(rng: random.Random, pools: Pools) -> str:
    if rng.random() < 0.5:
        return _prefix(rng, rng.choice(pools.bait))
    c = rng.choice(list(COVERED.values()))
    metric = rng.choice(["total revenue", "net income", "research and development spending", "headcount", "capital expenditures"])
    return _prefix(rng, f"What was {c}'s {metric} in {rng.choice(BAIT_YEARS)}?")


def _ramp(day: int, start: int, end_day: int, lo: float, hi: float) -> float:
    if day < start:
        return 0.0
    return lo + (hi - lo) * (day - start) / max(end_day - start, 1)


def question_for(scenario: str, day: int, n_days: int, rng: random.Random, pools: Pools) -> str:
    """Baseline traffic carries a small background of off-topic questions and attacks."""
    shift_start = n_days // 2
    last = n_days - 1
    r = rng.random()
    if scenario == "topic_shift":
        p_other = _ramp(day, shift_start, last, 0.15, 0.55)
        p_general = _ramp(day, shift_start, last, 0.05, 0.15)
        if r < p_other:
            return other_company_question(rng)
        if r < p_other + p_general:
            return rng.choice(GENERAL)
    elif scenario == "attack_wave":
        if day == last and r < 0.35:
            return rng.choice(pools.attacks)
    elif scenario == "hallucination_pressure":
        if r < _ramp(day, shift_start, last, 0.15, 0.55):
            return bait_question(rng, pools)
    r2 = rng.random()
    if r2 < 0.03:
        return rng.choice(GENERAL)
    if r2 < 0.05:
        return rng.choice(pools.attacks)
    return baseline_question(rng, pools)


def degraded_from(scenario: str, n_days: int) -> int | None:
    return n_days // 2 + 2 if scenario == "model_drift" else None


def simulate(scenario: str, start: date, n_days: int = 14, per_day: int = 60, seed: int = 0, service=None,
             target: str | None = None, rps: float = 2.0, golden_limit: int = 30, golden_history: str | None = None,
             env: str | None = None, on_day: Callable[[int, date], None] | None = None) -> dict:
    """Generate ``n_days`` of backdated traffic for ``scenario``. Returns a small summary."""
    from monitor.golden_run import append_history
    from monitor.golden_run import run as run_golden
    from rag.llm import ExtractiveGenerator, FaultInjectingGenerator

    rng = random.Random(seed)
    pools = Pools.load()
    env = env or f"sim-{scenario.replace('_', '-')}"
    degrade_day = degraded_from(scenario, n_days)
    sent = 0
    original = None
    if service is not None:
        original = (service.c.generator, service.c.alternate)
    client = httpx.Client(base_url=target.rstrip("/"), timeout=120) if target else None
    try:
        for day in range(n_days):
            d = start + timedelta(days=day)
            if service is not None and degrade_day is not None and day >= degrade_day:
                fault = FaultInjectingGenerator(ExtractiveGenerator(), number_rate=0.5, citation_drop_rate=0.5, seed=seed + day)
                service.c.generator = service.c.alternate = fault
            for i in range(per_day):
                q = question_for(scenario, day, n_days, rng, pools)
                ts = datetime(d.year, d.month, d.day, tzinfo=UTC) + timedelta(seconds=int(86400 * (i + rng.random()) / per_day))
                if client is not None:
                    headers = {"X-Trace-Env": env, "X-Trace-Timestamp": ts.isoformat(), "X-Sim-Secret": os.environ.get("SIM_SECRET", "")}
                    client.post("/ask", json={"question": q, "include_guards": False}, headers=headers).raise_for_status()
                    time.sleep(1.0 / rps)
                else:
                    service.ask(q, environment=env, timestamp=ts, tags=[f"scenario:{scenario}"])
                sent += 1
            if golden_history and service is not None:
                res = run_golden("inprocess", service=service, limit=golden_limit)
                res["timestamp"] = (datetime(d.year, d.month, d.day, tzinfo=UTC) + timedelta(hours=23, minutes=30)).isoformat()
                res["simulated_day"] = day
                append_history(res, golden_history)
            if on_day:
                on_day(day, d)
    finally:
        if client is not None:
            client.close()
        if service is not None and original is not None:
            service.c.generator, service.c.alternate = original
            service.c.tracer.flush()
    return {"scenario": scenario, "env": env, "requests": sent, "start": str(start), "days": n_days,
            "degraded_from_day": degrade_day}


def main() -> None:
    from rag.service import RAGService

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=["baseline", "topic_shift", "attack_wave", "hallucination_pressure", "model_drift"])
    ap.add_argument("--start", default=None, help="first simulated day (default: 14 days ago)")
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--per-day", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", default=None, help="deployed base URL; omit to run in-process")
    ap.add_argument("--rps", type=float, default=1.0, help="request rate against --target (stay inside free tiers)")
    ap.add_argument("--golden-history", default=None)
    args = ap.parse_args()
    start = date.fromisoformat(args.start) if args.start else (datetime.now(UTC) - timedelta(days=args.days)).date()
    svc = None if args.target else RAGService.build()
    if svc is not None:
        from rag.tracing import LocalBackend

        if isinstance(svc.c.tracer.backend, LocalBackend):
            svc.c.tracer.backend.clear(f"sim-{args.scenario.replace('_', '-')}")
    print(json.dumps(simulate(args.scenario, start, args.days, args.per_day, args.seed, svc, args.target, args.rps,
                              golden_history=args.golden_history), indent=2))


if __name__ == "__main__":
    main()
