from __future__ import annotations

import logging

import pytest

logging.getLogger("rag.embedder").setLevel(logging.ERROR)
logging.getLogger("guards.injection").setLevel(logging.ERROR)
logging.getLogger("guards.nli").setLevel(logging.ERROR)


class ScriptedGenerator:
    """Returns canned answers in order and records every request (to inspect what reached the 'LLM')."""

    provider, model = "scripted", "scripted-v1"

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.requests = []

    def generate(self, req):
        from rag.llm import GenerationResult

        self.requests.append(req)
        text = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return GenerationResult(text, self.provider, self.model, 1.0)


@pytest.fixture(scope="session")
def service():
    from rag.llm import ExtractiveGenerator
    from rag.service import RAGService

    return RAGService.build(guards_config="configs/guards/v2.yaml", generator=ExtractiveGenerator(), trace_backend="none")


@pytest.fixture(scope="session")
def pipeline(service):
    return service.c.pipeline


@pytest.fixture()
def scripted(service):
    """Swap in a scripted generator for one test, restore afterwards."""
    original = (service.c.generator, service.c.alternate)

    def _use(answers: list[str]) -> ScriptedGenerator:
        g = ScriptedGenerator(answers)
        service.c.generator = service.c.alternate = g
        return g

    yield _use
    service.c.generator, service.c.alternate = original
