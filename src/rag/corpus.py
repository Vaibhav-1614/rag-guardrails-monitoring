"""Load the vendored 10-K chunk artifacts (data/index) into memory."""
from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property, lru_cache

import numpy as np

from rag.config import load_settings, resolve
from rag.text import tokenize


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    ticker: str
    section: str
    text: str
    index: int  # row in the corpus matrices


@dataclass
class Corpus:
    chunks: list[Chunk]
    chunk_size: int
    filings: dict[str, dict] = field(default_factory=dict)

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
    def tokenized(self) -> list[list[str]]:
        return [tokenize(c.text) for c in self.chunks]

    @cached_property
    def table_scale(self) -> dict[str, float]:
        """Dominant '(in millions|thousands)' table convention per ticker, for unitless table numbers."""
        out: dict[str, float] = {}
        for t in sorted(set(self.tickers.tolist())):
            text = " ".join(c.text for c in self.chunks if c.ticker == t).lower()
            counts = {
                1e6: len(re.findall(r"in millions", text)),
                1e3: len(re.findall(r"in thousands", text)),
                1e9: len(re.findall(r"in billions", text)),
            }
            out[t] = max(counts, key=counts.get) if any(counts.values()) else 1e6
        return out

    @cached_property
    def fiscal_years(self) -> dict[str, int]:
        """Fiscal year each filing reports on, from the period date in its document name (aapl-20250927.htm)."""
        out: dict[str, int] = {}
        for t, meta in self.filings.items():
            m = re.search(r"(20\d{2})(\d{2})(\d{2})", meta.get("primaryDocument", ""))
            if m:
                out[t] = int(m.group(1))
            elif meta.get("filingDate"):
                y, mo = int(meta["filingDate"][:4]), int(meta["filingDate"][5:7])
                out[t] = y - 1 if mo <= 4 else y
        return out

    def mask(self, tickers: list[str] | tuple[str, ...] | None) -> np.ndarray:
        if not tickers:
            return np.ones(len(self.chunks), dtype=bool)
        return np.isin(self.tickers, list(tickers))

    def __len__(self) -> int:
        return len(self.chunks)


@lru_cache(maxsize=2)
def _load(index_dir: str, chunk_size: int) -> Corpus:
    base = resolve(index_dir)
    files = sorted((base / "chunks").glob(f"*_{chunk_size}.json.gz"))
    if not files:
        raise FileNotFoundError(f"No chunk artifacts in {base / 'chunks'} for chunk size {chunk_size}")
    chunks: list[Chunk] = []
    for f in files:
        with gzip.open(f, "rb") as fh:
            rows = json.loads(fh.read().decode("utf-8"))
        for r in rows:
            chunks.append(Chunk(r["chunk_id"], r["ticker"].upper(), r.get("section", ""), r["text"], len(chunks)))
    filings = {}
    manifest = base / "MANIFEST.json"
    if manifest.exists():
        filings = json.loads(manifest.read_text()).get("filings", {})
    return Corpus(chunks=chunks, chunk_size=chunk_size, filings=filings)


def load_corpus(settings: dict | None = None) -> Corpus:
    settings = settings or load_settings()
    c = settings.get("corpus", {})
    return _load(c.get("index_dir", "data/index"), int(c.get("chunk_size", 256)))


def chunk_files_order(settings: dict | None = None) -> list[str]:
    """Ticker order of the artifact files (embedding matrices are concatenated in this order)."""
    settings = settings or load_settings()
    c = settings.get("corpus", {})
    base = resolve(c.get("index_dir", "data/index"))
    return [f.name.split("_")[0] for f in sorted((base / "chunks").glob(f"*_{int(c.get('chunk_size', 256))}.json.gz"))]
