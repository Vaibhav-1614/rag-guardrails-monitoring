"""Vendor the 10-K chunk artifacts from the rag-eval-platform repo and tag each chunk with its 10-K section.

    git clone --depth 1 https://github.com/Vaibhav-1614/rag-eval-platform ../rag-eval-platform
    python scripts/vendor_corpus.py --source ../rag-eval-platform --chunk-size 256

Writes data/index/chunks/{TICKER}_{size}.json.gz (+ "section"), copies the aligned
bge-small float16 embeddings, and records provenance in data/index/MANIFEST.json.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "XOM"]
HEADING = re.compile(r"(?m)^(?:Item|ITEM)\s?(\d{1,2}[A-C]?)\.(?:\s+(\S.*))?$")


def derive_sections(chunks: list[dict]) -> list[dict]:
    """Section in effect at each chunk, from the 'Item N.' headings seen so far.

    Table-of-contents chunks (many bare headings) are labelled 'Contents' and do not
    move the running section forward.
    """
    current = "Cover"
    out = []
    for c in chunks:
        heads = HEADING.findall(c["text"])
        titled = [(n, t) for n, t in heads if t and not re.fullmatch(r"[\d\s]+", t)]
        if len(heads) >= 5 and len(titled) < 3:
            section = "Contents"
        else:
            first_line_head = HEADING.match(c["text"].lstrip())
            section = f"Item {first_line_head.group(1)}" if first_line_head else current
            if titled:
                current = f"Item {titled[-1][0]}"
            elif heads and len(heads) < 5:
                current = f"Item {heads[-1][0]}"
        out.append({**c, "section": section})
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--chunk-size", type=int, default=256)
    args = ap.parse_args()
    src = args.source.resolve()
    dst = ROOT / "data" / "index"
    (dst / "chunks").mkdir(parents=True, exist_ok=True)
    (dst / "embeddings").mkdir(parents=True, exist_ok=True)
    manifest: dict = {"source_repo": "https://github.com/Vaibhav-1614/rag-eval-platform", "chunk_size": args.chunk_size, "files": {}}
    try:
        manifest["source_commit"] = subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        manifest["source_commit"] = None
    filings = {}
    for t in TICKERS:
        with gzip.open(src / "data/index/chunks" / f"{t}_{args.chunk_size}.json.gz", "rb") as f:
            chunks = json.loads(f.read().decode("utf-8"))
        chunks = derive_sections(chunks)
        out = dst / "chunks" / f"{t}_{args.chunk_size}.json.gz"
        with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as f:
            f.write(json.dumps(chunks, ensure_ascii=False).encode("utf-8"))
        emb_src = src / "data/index/embeddings" / f"{t}_{args.chunk_size}_bge.npy"
        emb_dst = dst / "embeddings" / emb_src.name
        shutil.copyfile(emb_src, emb_dst)
        meta_path = src / "data/raw" / t / "filing_meta.json"
        if meta_path.exists():
            filings[t] = json.loads(meta_path.read_text())
        manifest["files"][out.name] = {"chunks": len(chunks), "sha256": sha256(out)}
        manifest["files"][emb_dst.name] = {"sha256": sha256(emb_dst)}
        secs = sorted({c["section"] for c in chunks})
        print(f"{t}: {len(chunks)} chunks, sections={len(secs)}")
    manifest["filings"] = filings
    (dst / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
