"""The upstream v2 index (section- and table-aware 10-K chunks) loaded into memory.

Retrieval itself is done by the upstream pipeline; the guards use this view of the corpus for
scope centroids, the domain word model, table units, fiscal years and the sanitizer's corpus scan.
"""
from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property, lru_cache

import numpy as np

from rag.config import load_settings
from rag.upstream import upstream_root

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "XOM"]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    ticker: str
    section: str        # "Item 7"
    text: str
    index: int          # row in the corpus matrices
    kind: str = "text"  # "text" | "table"
    context: str = ""   # "<company> 10-K, fiscal <year> | <item> | <heading>"
    fiscal_year: int | None = None


@dataclass
class Corpus:
    chunks: list[Chunk]
    version: str = "v2"
    extra: dict = field(default_factory=dict)

    @cached_property
    def tickers(self) -> np.ndarray:
        return np.array([c.ticker for c in self.chunks])

    @cached_property
    def by_id(self) -> dict[str, Chunk]:
        return {c.chunk_id: c for c in self.chunks}

    @cached_property
    def token_counts(self) -> Counter:
        """Corpus unigram counts (used by the scope guard's domain score)."""
        cnt: Counter = Counter()
        for c in self.chunks:
            cnt.update(re.findall(r"[a-z]+", c.text.lower()))
        return cnt

    @cached_property
    def table_scale(self) -> dict[str, float]:
        """Dominant '(in millions|thousands)' table convention per ticker, for unitless table numbers."""
        out: dict[str, float] = {}
        for t in sorted(set(self.tickers.tolist())):
            text = " ".join(c.text for c in self.chunks if c.ticker == t).lower()
            counts = {
                1e6: len(re.findall(r"in millions|millions of dollars", text)),
                1e3: len(re.findall(r"in thousands", text)),
                1e9: len(re.findall(r"in billions", text)),
            }
            out[t] = max(counts, key=counts.get) if any(counts.values()) else 1e6
        return out

    @cached_property
    def fiscal_years(self) -> dict[str, int]:
        """Fiscal year each filing reports on (from the index metadata)."""
        out: dict[str, int] = {}
        for c in self.chunks:
            if c.fiscal_year and c.ticker not in out:
                out[c.ticker] = int(c.fiscal_year)
        return out

    def __len__(self) -> int:
        return len(self.chunks)


def _section_name(sec: str) -> str:
    if not sec:
        return "Other"
    return f"Item {sec}" if sec[0].isdigit() else sec.capitalize()  # "7" -> "Item 7", "cover" -> "Cover"


@lru_cache(maxsize=2)
def _load(root: str) -> Corpus:
    base = upstream_root() / "data" / "index" / "v2" / "chunks"
    chunks: list[Chunk] = []
    for t in TICKERS:
        with gzip.open(base / f"{t}.json.gz", "rb") as fh:
            rows = json.loads(fh.read().decode("utf-8"))
        for r in rows:
            sec = str(r.get("section") or "")
            chunks.append(Chunk(r["chunk_id"], r["ticker"].upper(), _section_name(sec), r["text"],
                                len(chunks), r.get("kind", "text"), r.get("context", ""), r.get("fiscal_year")))
    return Corpus(chunks=chunks)


def load_corpus(settings: dict | None = None) -> Corpus:
    settings = settings or load_settings()
    return _load(str(upstream_root(settings)))
