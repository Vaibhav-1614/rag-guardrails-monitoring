"""Versioned reference window (design doc §8.2): query texts + score distributions.

The reference is refreshed deliberately, never automatically, or drift would be absorbed silently.
Each refresh gets a new version file and a line in monitor/reference/CHANGELOG.md.

    python -m monitor.reference build --env sim-reference --start 2026-09-01 --end 2026-09-08 --out monitor/reference/v1.npz
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from monitor.features import to_row
from rag.config import resolve


@dataclass
class Reference:
    version: str
    created: str
    description: str
    texts: list[str]
    scope_scores: np.ndarray
    retrieval_scores: np.ndarray
    out_of_scope_rate: float
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)

    def embed(self, embedder) -> np.ndarray:
        """Embeddings under the active embedder (computed from texts if the file has none for it)."""
        if embedder.name not in self.embeddings:
            self.embeddings[embedder.name] = embedder.encode(self.texts).astype(np.float32)
        return self.embeddings[embedder.name]

    def save(self, path: str | Path) -> Path:
        p = resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = {"version": self.version, "created": self.created, "description": self.description,
                "out_of_scope_rate": self.out_of_scope_rate, "texts": self.texts}
        arrays = {f"emb_{k}": v for k, v in self.embeddings.items()}
        np.savez_compressed(p, meta=np.array(json.dumps(meta)), scope_scores=self.scope_scores,
                            retrieval_scores=self.retrieval_scores, **arrays)
        return p

    @classmethod
    def load(cls, path: str | Path) -> Reference:
        z = np.load(resolve(path), allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        embs = {k[4:]: z[k] for k in z.files if k.startswith("emb_")}
        return cls(meta["version"], meta["created"], meta["description"], meta["texts"], z["scope_scores"],
                   z["retrieval_scores"], float(meta["out_of_scope_rate"]), embs)


def build(records: list[dict], golden_questions: list[str], version: str, description: str, embedder=None) -> Reference:
    rows = [to_row(r) for r in records]
    texts = golden_questions + [r["question"] for r in rows if r["question"]]
    ref = Reference(
        version=version,
        created=datetime.now(UTC).strftime("%Y-%m-%d"),
        description=description,
        texts=texts,
        scope_scores=np.array([r["scope"] for r in rows if r["scope"] is not None], dtype=np.float32),
        retrieval_scores=np.array([r["retrieval_top"] for r in rows if r["retrieval_top"] is not None], dtype=np.float32),
        out_of_scope_rate=float(np.mean([r["scope_reason"] is not None for r in rows])) if rows else 0.0,
    )
    if embedder is not None:
        ref.embed(embedder)
    return ref


def main() -> None:
    from rag.config import load_settings
    from rag.embedder import get_embedder
    from rag.tracing import make_backend

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--env", required=True)
    b.add_argument("--start", required=True)
    b.add_argument("--end", required=True)
    b.add_argument("--source", default="auto")
    b.add_argument("--version", default="v1")
    b.add_argument("--out", default="monitor/reference/v1.npz")
    b.add_argument("--description", default="")
    args = ap.parse_args()
    settings = load_settings()
    backend = make_backend(settings, None if args.source == "auto" else args.source)
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC) + timedelta(days=1)
    records = backend.read(start, end, args.env)
    golden = [json.loads(line)["question"] for line in resolve("eval/golden.jsonl").read_text().splitlines() if line.strip()]
    ref = build(records, golden, args.version, args.description or f"golden set + {args.env} {args.start}..{args.end}", get_embedder(settings))
    p = ref.save(args.out)
    print(f"reference {args.version}: {len(ref.texts)} queries ({len(records)} traces) -> {p}")


if __name__ == "__main__":
    main()
