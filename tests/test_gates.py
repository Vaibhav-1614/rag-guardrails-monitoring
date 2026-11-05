"""'A PR with a weakened guard gets blocked' (doc §12, week 3)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from run_guard_eval import Evaluator, check_gates  # noqa: E402


@pytest.fixture(scope="module")
def evaluator():
    return Evaluator("configs/guards/v2.yaml", "extractive", "all")


def _gate(report, name):
    return next(g for g in check_gates(report, ROOT / "eval" / "gates.yaml") if g["gate"] == name)


def test_weakened_injection_guard_fails_its_gate(evaluator):
    guard = evaluator.p.guards["prompt_injection"]
    evaluator.eval_injection()
    rep = evaluator.report
    rep.setdefault("system", {})["injection_asr_enforced"] = 1 - rep["guards"]["prompt_injection"]["recall"]
    assert _gate(rep, "injection_asr")["status"] == "pass"
    original = guard.threshold
    guard.threshold = 0.99  # "weakened" guard: only near-certain rule hits block
    try:
        evaluator.eval_injection()
        rep["system"]["injection_asr_enforced"] = 1 - rep["guards"]["prompt_injection"]["recall"]
        assert _gate(rep, "injection_asr")["status"] == "fail"
    finally:
        guard.threshold = original


def test_loosened_numeric_tolerance_fails_its_gate(evaluator):
    checker = evaluator.p.guards["numeric_grounding"].checker
    evaluator.eval_numeric()
    assert _gate(evaluator.report, "numeric_recall")["status"] == "pass"
    original = checker.rel_tol
    checker.rel_tol = 0.01
    try:
        evaluator.eval_numeric()
        assert _gate(evaluator.report, "numeric_recall")["status"] == "fail"
    finally:
        checker.rel_tol = original
