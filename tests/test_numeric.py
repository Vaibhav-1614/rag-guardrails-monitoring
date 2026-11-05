import pytest

from guards.numeric import NumericChecker, extract_quantities

CTX = [{"ticker": "AAPL", "text": "(In millions)\nTotal net sales $ 416,161 $ 391,035 $ 383,285\nNet income 112,010 93,736\n"
                                   "Gross margin percentage 46.9 % 46.2 %"}]


def kinds(text):
    return [(q.text, q.kind, q.value) for q in extract_quantities(text)]


def test_extracts_units_scale_and_negatives():
    q = {t: (k, v) for t, k, v in kinds("Sales were $94.3 billion, margin 12.5%, a loss of (1,234) and 1,234 million; $5.2bn.")}
    assert q["$94.3 billion"] == ("currency", pytest.approx(94.3e9))
    assert q["12.5%"] == ("percent", pytest.approx(0.125))
    assert q["(1,234)"] == ("number", -1234)
    assert q["1,234 million"] == ("number", pytest.approx(1.234e9))
    assert q["$5.2bn"] == ("currency", pytest.approx(5.2e9))


def test_skips_identifiers_dates_and_citations():
    found = [t for t, _, _ in kinds("See Item 7 and Note 12 of the Form 10-K, 401(k), H100, Q3, COVID-19, "
                                    "on September 27, 2025 [3], with 3 segments.")]
    assert found == ["2025"]


@pytest.mark.parametrize("answer", [
    "Total net sales were $416,161 million [1].",
    "Total net sales were $416.2 billion [1].",      # honest rounding
    "Total net sales were $416 billion [1].",        # coarser rounding
    "Total net sales were $416.1 billion [1].",      # truncation
    "Total net sales were about $0.4 trillion [1].",  # very coarse, still consistent
])
def test_supported_figures(answer):
    checks = NumericChecker().check(answer, CTX, table_scale={"AAPL": 1e6})
    assert all(c.status == "supported" for c in checks), [(c.quantity.text, c.status) for c in checks]


@pytest.mark.parametrize("answer", [
    "Total net sales were $418.2 billion [1].",   # 0.5% off: passes the doc's 0.5% tolerance, caught here
    "Total net sales were $94.8 billion [1].",
    "Total net sales were $416,161 billion [1].",  # scale error
    "Net income was 112,910 [1].",                 # one wrong digit
])
def test_unsupported_figures(answer):
    checks = NumericChecker(rel_tol=0.0005).check(answer, CTX, table_scale={"AAPL": 1e6})
    assert any(c.status == "unsupported" for c in checks)


def test_loose_tolerance_misses_near_miss():
    checks = NumericChecker(rel_tol=0.005).check("Total net sales were $418.2 billion [1].", CTX, table_scale={"AAPL": 1e6})
    assert all(c.status == "supported" for c in checks)


def test_derivations_from_table_row_and_sentence():
    ck = NumericChecker()
    pct = ck.check("Net income grew 19.5% [1].", CTX, table_scale={"AAPL": 1e6})
    assert pct[0].status == "derived" and pct[0].how.startswith("percent_change")
    diff = ck.check("Net sales increased by $25,126 million, from $391,035 million to $416,161 million [1].", CTX, table_scale={"AAPL": 1e6})
    assert [c.status for c in diff] == ["derived", "supported", "supported"]
    wrong = ck.check("Net income grew 25.0% [1].", CTX, table_scale={"AAPL": 1e6})
    assert wrong[0].status == "unsupported"


def test_years_in_reporting_window_and_question_are_ok():
    ck = NumericChecker()
    ok = ck.check("In fiscal 2025 net sales were $416,161 million [1].", CTX, table_scale={"AAPL": 1e6}, period_years={2023, 2024, 2025})
    assert all(c.status == "supported" for c in ok)
    bad = ck.check("In fiscal 2019 net sales were $416,161 million [1].", CTX, table_scale={"AAPL": 1e6}, period_years={2023, 2024, 2025})
    assert bad[0].status == "unsupported"
    exempt = ck.check("Yes, above $400 billion [1].", CTX, question="Were sales above $400 billion?")
    assert exempt[0].status == "exempt"


def test_miscited_number_is_flagged():
    chunks = [{"ticker": "AAPL", "text": "Revenue was $10 million."}, {"ticker": "AAPL", "text": "Costs were $7.5 million."}]
    checks = NumericChecker().check("Costs were $7.5 million [1].", chunks)
    assert checks[0].status == "miscited"
