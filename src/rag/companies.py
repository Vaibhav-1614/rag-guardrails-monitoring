"""Company registry: which companies the corpus covers, and detection of company mentions."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from rag.config import load_settings, load_yaml


@dataclass(frozen=True)
class CompanyMentions:
    covered: tuple[str, ...] = ()          # tickers in the corpus, in order of first mention
    other: tuple[str, ...] = ()            # names/tickers of companies outside the corpus
    spans: tuple[tuple[int, int], ...] = ()  # character spans of every mention

    @property
    def unknown_only(self) -> bool:
        return bool(self.other) and not self.covered


@dataclass
class CompanyRegistry:
    covered: dict[str, dict]
    other_names: list[str]
    other_tickers: list[str]
    _alias_re: re.Pattern = field(init=False, repr=False)
    _alias_map: dict[str, str] = field(init=False, repr=False)
    _ticker_re: re.Pattern = field(init=False, repr=False)
    _other_name_re: re.Pattern | None = field(init=False, repr=False)
    _other_ticker_re: re.Pattern | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._alias_map = {}
        for ticker, info in self.covered.items():
            for alias in [info["name"], *info.get("aliases", []), *info.get("subsidiaries", [])]:
                self._alias_map[alias.lower()] = ticker
        aliases = sorted(self._alias_map, key=len, reverse=True)
        self._alias_re = re.compile(r"(?<![\w&])(" + "|".join(re.escape(a) for a in aliases) + r")(?![\w&])", re.I)
        self._ticker_re = re.compile(r"(?<![\w$])\$?(" + "|".join(map(re.escape, self.covered)) + r")(?![\w])")
        names = sorted({n.lower() for n in self.other_names}, key=len, reverse=True)
        self._other_name_re = (
            re.compile(r"(?<![\w&])(" + "|".join(re.escape(n) for n in names) + r")(?![\w&])", re.I) if names else None
        )
        self._other_ticker_re = (
            re.compile(r"(?<![\w$])\$?(" + "|".join(map(re.escape, self.other_tickers)) + r")(?![\w])")
            if self.other_tickers
            else None
        )

    @property
    def tickers(self) -> list[str]:
        return list(self.covered)

    def name(self, ticker: str) -> str:
        return self.covered[ticker]["name"]

    def all_alias_tokens(self) -> frozenset[str]:
        """Single tokens of covered-company aliases (dropped from retrieval-confidence terms)."""
        toks: set[str] = set()
        for info in self.covered.values():
            for alias in [info["name"], *info.get("aliases", [])]:
                toks.update(re.findall(r"[a-z0-9]+", alias.lower()))
        toks.update(t.lower() for t in self.covered)
        return frozenset(toks)

    def detect(self, text: str) -> CompanyMentions:
        found: list[tuple[int, str]] = []
        spans: list[tuple[int, int]] = []
        for m in self._alias_re.finditer(text):
            found.append((m.start(), self._alias_map[m.group(1).lower()]))
            spans.append(m.span())
        for m in self._ticker_re.finditer(text):
            found.append((m.start(), m.group(1)))
            spans.append(m.span())
        covered: list[str] = []
        for _, t in sorted(found):
            if t not in covered:
                covered.append(t)
        other: list[str] = []
        covered_spans = list(spans)

        def overlaps(span: tuple[int, int]) -> bool:
            return any(not (span[1] <= a or span[0] >= b) for a, b in covered_spans)

        for rx in (self._other_name_re, self._other_ticker_re):
            if rx is None:
                continue
            for m in rx.finditer(text):
                if overlaps(m.span()):
                    continue
                label = m.group(1)
                if label.lower() not in [o.lower() for o in other]:
                    other.append(label)
                spans.append(m.span())
        return CompanyMentions(tuple(covered), tuple(other), tuple(sorted(spans)))


@lru_cache(maxsize=4)
def _registry(path: str) -> CompanyRegistry:
    cfg = load_yaml(path)
    other = cfg.get("other", {}) or {}
    return CompanyRegistry(
        covered=cfg["covered"],
        other_names=list(other.get("names", [])),
        other_tickers=list(other.get("tickers", [])),
    )


def get_registry(settings: dict | None = None) -> CompanyRegistry:
    settings = settings or load_settings()
    return _registry(settings.get("companies_config", "configs/companies.yaml"))
