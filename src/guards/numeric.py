"""§6.4 Numeric grounding (T6): every number in the answer must come from the cited filing text.

1. Extract numbers with unit and scale: "$94.3 billion", "12.5%", "1,234 million", "(1,234)".
2. Normalise to canonical values: 94.3e9 (currency), 0.125 (percent), negatives from parentheses.
3. Extract numbers from the *cited* chunks the same way. Unitless table figures may be stated in
   the table's unit ("in millions"), so they are also tried at the chunk's / filing's table scale.
4. An answer number is supported when some context number matches within tolerance:
   ``|ctx - ans| <= max(rel_tol * |ans|, half a unit of the answer's last shown digit)``.
   Precision-aware slack accepts honest rounding ("$416,161 million" -> "$416.2 billion" or
   "$416 billion") while a wrong digit ("$94.8B" vs "$94.3B") still fails.
   It is also supported when it is derivable by an operation the sentence states (percent change,
   difference, sum, share) from two numbers in the same answer sentence or the same table row.
5. Unsupported numbers fail the guard -> regenerate, then flag.

Identifiers are not quantities and are skipped: "Item 7", "Note 12", "Form 10-K", "401(k)",
"H100", "Q3", "COVID-19", "ASU 2023-09", day-of-month in dates, citation markers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from guards.base import BaseGuard, GuardContext
from guards.citation import CITE, cited_ids
from rag.text import split_sentences

SCALE_WORDS = {
    "trillion": 1e12, "tn": 1e12, "billion": 1e9, "bn": 1e9, "bil": 1e9, "million": 1e6, "mn": 1e6,
    "mm": 1e6, "mil": 1e6, "thousand": 1e3,
}
SCALE_LETTERS = {"t": 1e12, "b": 1e9, "m": 1e6, "k": 1e3}
MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december"
ID_KEYWORDS = re.compile(
    r"(?:item|items|note|notes|part|rule|rules|section|sections|topic|asu|form|no\.|nos\.|number|article|exhibit|schedule|"
    r"level|tier|phase|model|series|class|chapter|title|regulation|page|pages|step|figure|table|q|h|fy)\s*$", re.I)

_NUM_RE = re.compile(
    r"(?P<open>\()?"
    r"(?P<cur>US\$|USD\s?|\$|€|£)?\s?"
    r"(?P<sign>[-−–])?"
    r"(?P<cur2>\$)?\s?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?P<close>\))?"
    r"(?:(?P<letter>[TtBbMmKk])(?![A-Za-z])|\s?(?P<word>trillion|billion|million|thousand|tn|bn|bil|mn|mm|mil)\b)?"
    r"(?:\s?(?P<pct>%|percent\b|per\s?cent\b|percentage\s+points?\b|basis\s+points?\b|bps\b|pp\b))?"
)


@dataclass(frozen=True)
class Quantity:
    text: str
    value: float          # canonical: dollars / plain value / fraction for percent / year
    mantissa: float       # the number as written (before scale), signed
    scale: float          # multiplier from scale word/letter (1 if none)
    kind: str             # currency | number | percent | bps | year
    decimals: int
    explicit_scale: bool
    start: int
    end: int
    line: int = 0         # line index in its source text (table row)

    @property
    def unit(self) -> float:
        """Size of one unit in the last shown digit, in canonical units (trailing zeros = coarse)."""
        digits = re.sub(r"[^\d.]", "", self.text.split("%")[0])
        int_part = digits.split(".")[0] if "." in digits else digits
        if self.decimals == 0:
            trailing = len(int_part) - len(int_part.rstrip("0")) if int_part.strip("0") else 0
            base = 10.0 ** min(trailing, max(len(int_part) - 1, 0))
        else:
            base = 10.0 ** (-self.decimals)
        mult = self.scale / 100.0 if self.kind == "percent" else self.scale
        if self.kind == "bps":
            mult = self.scale
        return base * mult


def _is_identifier(text: str, start: int, end: int, m: re.Match) -> bool:
    before = text[max(0, start - 14):start]
    after = text[end:end + 3]
    if m.group("cur") or m.group("cur2") or m.group("pct") or m.group("word"):
        # Money/percent are quantities even after a keyword ("Note 5 ... $3.2 billion")
        return False
    if start > 0 and (text[start - 1].isalpha() or text[start - 1] in "_#"):
        return True  # H100, Q3, B200, #5
    if re.search(r"[A-Za-z]-$", before):
        return True  # COVID-19, Nasdaq-100
    if re.match(r"-[A-Za-z]|\(\s?[a-z]\s?\)|[A-Za-z]", after):
        return True  # 10-K, 401(k), 5G
    if ID_KEYWORDS.search(before):
        return True
    if re.search(rf"\b(?:{MONTHS})\.?\s*$", before, re.I):
        return True  # day of month: "September 27"
    return bool(re.match(r"-\d{2}\b", after) and re.search(r"(?:asu|no\.)\s*$", before, re.I))


def extract_quantities(text: str, ignore_small_integers: int = 12) -> list[Quantity]:
    """Extract quantities. Citation markers are blanked (offsets preserved) first."""
    clean = CITE.sub(lambda m: " " * len(m.group(0)), text)
    line_starts = [0] + [m.end() for m in re.finditer(r"\n", clean)]
    out: list[Quantity] = []
    for m in _NUM_RE.finditer(clean):
        num = m.group("num")
        start, end = m.start("num"), m.end()
        if _is_identifier(clean, m.start("num"), m.end("num"), m):
            continue
        letter = m.group("letter")
        cur = m.group("cur") or m.group("cur2")
        if letter and not (cur or letter.lower() in "bmk" and re.match(r"\d", num)):
            letter = None
        if letter and letter.lower() == "t" and not cur:
            letter = None
        try:
            mant = float(num.replace(",", ""))
        except ValueError:
            continue
        neg = bool(m.group("sign")) or (bool(m.group("open")) and bool(m.group("close")))
        if neg:
            mant = -mant
        decimals = len(num.split(".")[1]) if "." in num else 0
        word = (m.group("word") or "").lower()
        scale = SCALE_WORDS.get(word, 1.0)
        if letter:
            scale = SCALE_LETTERS[letter.lower()]
        pct = (m.group("pct") or "").lower()
        if pct.startswith(("basis", "bps")):
            kind, value = "bps", mant
        elif pct:
            kind, value = "percent", mant * scale / 100.0
        elif cur:
            kind, value = "currency", mant * scale
        elif decimals == 0 and "," not in num and scale == 1.0 and 1900 <= abs(mant) <= 2100 and len(num) == 4:
            kind, value = "year", mant
        else:
            kind, value = "number", mant * scale
        if kind == "number" and scale == 1.0 and decimals == 0 and abs(mant) <= ignore_small_integers:
            continue  # small counts / ordinals ("3 segments", "2 reasons")
        line = max(i for i, s in enumerate(line_starts) if s <= start)
        surface = clean[m.start():end].strip()
        out.append(Quantity(surface, value, mant, scale, kind, decimals, bool(word or letter), start, end, line))
    return out


def _compatible(a: Quantity, c: Quantity) -> bool:
    if a.kind == "year":
        return c.kind in ("year", "number")
    if a.kind == "percent":
        return c.kind == "percent"
    if a.kind == "bps":
        return c.kind in ("bps", "percent")
    return c.kind in ("currency", "number")


def _context_values(c: Quantity, table_scales: tuple[float, ...]) -> list[float]:
    if c.kind == "percent":
        return [c.value]
    if c.kind == "bps":
        return [c.value]
    if c.kind == "year":
        return [c.value]
    if c.explicit_scale:
        return [c.value]
    # Unitless or bare-$ table figures: try as written and in the table's unit.
    return [c.mantissa * s for s in (1.0, *table_scales)]


def _close(ans: float, ctx: float, tol_abs: float, rel_tol: float) -> bool:
    a, b = abs(ans), abs(ctx)
    return abs(a - b) <= max(tol_abs, rel_tol * a) + 1e-12


@dataclass
class Evidence:
    quantities: list[Quantity]
    table_scales: tuple[float, ...]


@dataclass
class NumberCheck:
    quantity: Quantity
    status: str            # supported | derived | exempt | unsupported | miscited
    sentence: str
    how: str | None = None
    matched: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        q = self.quantity
        return {"text": q.text, "kind": q.kind, "value": q.value, "status": self.status, "how": self.how,
                "sentence": self.sentence[:200]}


_CHANGE_CUE = re.compile(r"\b(?:increase[ds]?|decrease[ds]?|grew|grow(?:th|n|s)?|declin(?:e|ed|es|ing)|rose|rise|fell|"
                         r"fall|drop(?:ped)?|up|down|change[ds]?|higher|lower|more|less|jump(?:ed)?|surge[ds]?|"
                         r"climb(?:ed)?|shr[ai]nk|reduc(?:ed|tion)|improv(?:ed|ement)|year[- ]over[- ]year|compared)\b", re.I)
_SUM_CUE = re.compile(r"\b(?:total(?:ed|ing|s)?|combined|together|sum|aggregate|in all|altogether)\b", re.I)
_SHARE_CUE = re.compile(r"\b(?:of (?:total|its|their|the)|share|accounted for|represent(?:ed|s|ing)?|portion|proportion|percent of|% of)\b", re.I)


class NumericChecker:
    def __init__(self, rel_tol: float = 0.001, ignore_small_integers: int = 12, require_cited_chunk: bool = True,
                 derivations: bool = True) -> None:
        self.rel_tol, self.ignore_small, self.require_cited, self.derivations = (
            rel_tol, ignore_small_integers, require_cited_chunk, derivations)

    def _evidence(self, chunks: list[dict], table_scale: dict[str, float]) -> list[Evidence]:
        ev = []
        for c in chunks:
            text = c["text"]
            scales = set()
            low = text.lower()
            for word, s in (("in millions", 1e6), ("in thousands", 1e3), ("in billions", 1e9)):
                if word in low:
                    scales.add(s)
            scales.add(table_scale.get(c.get("ticker", ""), 1e6))
            ev.append(Evidence(extract_quantities(text, ignore_small_integers=-1), tuple(sorted(scales))))
        return ev

    def _match(self, q: Quantity, evs: list[Evidence]) -> str | None:
        tol = q.unit / 2
        for ev in evs:
            for c in ev.quantities:
                if not _compatible(q, c):
                    continue
                for v in _context_values(c, ev.table_scales):
                    if q.kind == "year":
                        if int(abs(v)) == int(abs(q.value)) and c.decimals == 0:
                            return c.text
                    elif _close(q.value, v, tol, self.rel_tol) or (q.kind == "bps" and c.kind == "percent" and _close(q.value, v * 1e4, tol, self.rel_tol)) or q.kind in ("currency", "number", "percent") and abs(q.value) - 1e-12 <= abs(v) < abs(q.value) + q.unit:
                        return c.text
        return None

    def _derive(self, q: Quantity, sentence: str, operands: list[float]) -> str | None:
        if len(operands) < 2:
            return None
        tol = q.unit / 2
        change = bool(_CHANGE_CUE.search(sentence))
        total = bool(_SUM_CUE.search(sentence))
        share = bool(_SHARE_CUE.search(sentence))
        for a, b in combinations(operands, 2):
            if q.kind == "percent":
                for x, y in ((a, b), (b, a)):
                    if change and x != 0 and _close(q.value, (y - x) / abs(x), tol, self.rel_tol):
                        return "percent_change"
                    if share and y != 0 and _close(q.value, x / y, tol, self.rel_tol):
                        return "share"
            elif q.kind in ("currency", "number"):
                if change and _close(q.value, abs(b - a), tol, self.rel_tol):
                    return "difference"
                if total and _close(q.value, a + b, tol, self.rel_tol):
                    return "sum"
        return None

    def check(self, answer: str, chunks: list[dict], question: str = "", table_scale: dict[str, float] | None = None,
              period_years: set[int] | None = None) -> list[NumberCheck]:
        """``period_years``: fiscal years the cited filings report on; a year in this window is
        supported even when the table header row sits in a different chunk."""
        table_scale = table_scale or {}
        period_years = period_years or set()
        all_ev = self._evidence(chunks, table_scale)
        q_vals = extract_quantities(question, ignore_small_integers=-1) if question else []
        results: list[NumberCheck] = []
        for sent in split_sentences(answer):
            ids = [i for i in cited_ids(sent) if 1 <= i <= len(chunks)]
            cited_ev = [all_ev[i - 1] for i in ids] if ids else all_ev
            sent_checks: list[NumberCheck] = []
            for q in extract_quantities(sent, self.ignore_small):
                if any(_compatible(q, x) and _close(q.value, x.value, q.unit / 2, self.rel_tol) for x in q_vals):
                    sent_checks.append(NumberCheck(q, "exempt", sent, "in_question"))
                    continue
                if q.kind == "year" and int(q.value) in period_years:
                    sent_checks.append(NumberCheck(q, "supported", sent, "reporting_period"))
                    continue
                hit = self._match(q, cited_ev)
                if hit is not None:
                    sent_checks.append(NumberCheck(q, "supported", sent, "cited_chunk" if ids else "context", [hit]))
                    continue
                if ids and len(cited_ev) < len(all_ev):
                    other = self._match(q, all_ev)
                    if other is not None:
                        status = "miscited" if self.require_cited else "supported"
                        sent_checks.append(NumberCheck(q, status, sent, "uncited_chunk", [other]))
                        continue
                sent_checks.append(NumberCheck(q, "unsupported", sent))
            if self.derivations:
                self._apply_derivations(sent_checks, sent, cited_ev)
            results.extend(sent_checks)
        return results

    def _apply_derivations(self, checks: list[NumberCheck], sentence: str, evs: list[Evidence]) -> None:
        pending = [c for c in checks if c.status in ("unsupported", "miscited")]
        if not pending:
            return
        stated = [c.quantity.value for c in checks if c.status in ("supported", "exempt") and c.quantity.kind in ("currency", "number")]
        for chk in pending:
            how = self._derive(chk.quantity, sentence, stated)
            if how is None:
                # Same table row in a cited chunk: "Total net sales 416,161 391,035 ..." -> 6.4% growth
                for ev in evs:
                    rows: dict[int, list[float]] = {}
                    for c in ev.quantities:
                        if c.kind in ("currency", "number"):
                            rows.setdefault(c.line, []).append(c.value)
                    for vals in rows.values():
                        if 2 <= len(vals) <= 12:
                            how = self._derive(chk.quantity, sentence, vals)
                            if how:
                                how = f"{how}_table_row"
                                break
                    if how:
                        break
            if how:
                chk.status, chk.how = "derived", how


class NumericGroundingGuard(BaseGuard):
    name = "numeric_grounding"
    stage = "output"
    fail_action = "regenerate"

    def __init__(self, config: dict | None = None, table_scale: dict[str, float] | None = None,
                 fiscal_years: dict[str, int] | None = None) -> None:
        super().__init__(config)
        self.table_scale = table_scale or {}
        self.fiscal_years = fiscal_years or {}
        self.checker = NumericChecker(
            rel_tol=float(self.config.get("tolerance", 0.001)),
            ignore_small_integers=int(self.config.get("ignore_small_integers", 12)),
            require_cited_chunk=bool(self.config.get("require_cited_chunk", True)),
            derivations=bool(self.config.get("derivations", True)),
        )

    def evaluate(self, ctx: GuardContext) -> tuple[bool, float | None, dict[str, Any]]:
        years: set[int] = set()
        for t in {c.get("ticker") for c in ctx.chunks}:
            fy = self.fiscal_years.get(t or "")
            if fy:
                years.update(range(fy - 2, fy + 1))  # a 10-K reports three fiscal years
        checks = self.checker.check(ctx.answer or "", ctx.chunks, ctx.effective_question, self.table_scale, years)
        considered = [c for c in checks if c.status != "exempt"]
        bad = [c for c in considered if c.status in ("unsupported", "miscited")]
        score = 1.0 - len(bad) / len(considered) if considered else 1.0
        detail = {
            "n_numbers": len(considered),
            "n_supported": sum(c.status == "supported" for c in considered),
            "n_derived": sum(c.status == "derived" for c in considered),
            "unsupported": [c.to_dict() for c in bad],
            "numbers": [c.to_dict() for c in checks][:30],
        }
        if bad:
            detail["reasons"] = [f"number '{c.quantity.text}' is not in the cited chunks" + (" (found in another chunk)" if c.status == "miscited" else "") for c in bad[:5]]
        return not bad, float(score), detail
