"""The service runs on the upstream (doc-01) pipeline: facts table, v2 index, answerability."""
from rag.golden import answer_check, load_golden
from rag.llm import cite_every_sentence
from rag.text import split_sentences


def test_golden_set_comes_from_upstream_with_splits():
    g = load_golden()
    assert len(g) >= 176 and {x["split"] for x in g} == {"dev", "test"}
    assert sum(not x["answerable"] for x in g) >= 28
    assert all("question" in x and "tickers" in x for x in g)


def test_company_wide_figures_are_answered_from_the_facts_table(service):
    r = service.ask("How many people does Amazon employ?", trace=False)
    assert r.status == "answered" and "1,576,000" in r.answer and r.chunks[0].chunk_id == "AMZN_fact_employees"
    assert all(g.passed for g in r.guards if g.guard in ("citation", "numeric_grounding"))


def test_comparison_keeps_its_conclusion_and_cross_company_ranking_is_in_scope(service):
    r = service.ask("Who spends more on R&D, Apple or Microsoft?", trace=False)
    assert r.status == "answered" and "was higher" in r.answer and "stripped_uncited_sentences" not in r.flags
    r = service.ask("Which of the companies had the highest revenue?", trace=False)
    assert r.status == "answered" and "Amazon" in r.answer


def test_answerability_gate_uses_upstream_reasons(service):
    for q, reason in [("What is Apple's stock price today?", "market_data"),
                      ("What will NVIDIA's data center revenue be in fiscal 2029?", "future_period"),
                      ("What was Microsoft's revenue in fiscal 2015?", "period_not_in_corpus")]:
        r = service.ask(q, trace=False)
        gate = next(g for g in r.guards if g.guard == "retrieval_conf")
        assert r.refusal_reason == "retrieval_conf" and gate.detail["upstream_reason"] == reason, q


def test_cite_every_sentence():
    out = cite_every_sentence("A is $5 million in fiscal 2025 [1]. B is $7 million in fiscal 2025 [2]. B's was higher.")
    assert out.endswith("B's was higher [1][2].")
    out = cite_every_sentence("First sentence of the unit here. Second sentence of the unit here. [3]")
    assert out == "First sentence of the unit here [3]. Second sentence of the unit here [3]."
    assert len(split_sentences("Heading of a table [1] Results follow in the next unit.")) == 2


def test_answer_check():
    item = {"answer_contains": [["1,576,000", "1.58 million"]]}
    assert answer_check("Amazon employed 1,576,000 people [1].", item) and not answer_check("about 1.5 million", item)
    assert answer_check("anything", {"answer_contains": None}) is None
