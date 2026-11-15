"""Bridge to the doc-01 service: github.com/Vaibhav-1614/rag-eval-platform, pinned as a git
submodule at upstream/rag-eval-platform.

That repo owns the corpus (data/index/v2), the facts table, retrieval (stemmed BM25 + query
expansion, optional cross-encoder re-ranker), the answerability rules and the golden set. This
project wraps its pipeline with guards and monitoring; nothing is copied.

    git submodule update --init          # after cloning
    git submodule update --remote        # move to the latest upstream commit, then re-run `make eval`
"""
from __future__ import annotations

import logging
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from rag.config import ROOT, load_settings, resolve

logger = logging.getLogger(__name__)


def upstream_root(settings: dict | None = None) -> Path:
    settings = settings or load_settings()
    root = resolve(settings.get("upstream", {}).get("path", "upstream/rag-eval-platform"))
    if not (root / "src" / "rag_service").exists():
        raise FileNotFoundError(
            f"{root} is empty. Run `git submodule update --init` to fetch the rag-eval-platform service."
        )
    return root


def ensure_importable(settings: dict | None = None) -> Path:
    root = upstream_root(settings)
    for p in (str(root / "src"), str(root)):
        if p not in sys.path:
            sys.path.append(p)
    return root


def upstream_commit(settings: dict | None = None) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(upstream_root(settings)), "rev-parse", "--short=12", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


@lru_cache(maxsize=4)
def _pipeline(config: str, root: str):
    from rag_service.config import RetrievalConfig, ServiceSettings
    from rag_service.pipeline import Pipeline

    base = Path(root) / "configs" / "retrieval"
    candidates = ["v2.yaml", "v2-bm25.yaml"] if config == "auto" else [config]
    last: Exception | None = None
    for name in candidates:
        path = Path(name) if Path(name).is_absolute() else base / name
        try:
            cfg = RetrievalConfig.load(path)
            pipe = Pipeline(cfg, ServiceSettings.from_env())
            pipe.retriever.warm()  # loads the index and, for v2.yaml, the ONNX re-ranker
            return pipe
        except Exception as e:  # fastembed missing, model download blocked, ...
            last = e
            if config != "auto":
                raise
            logger.warning("upstream retrieval config %s unavailable (%s: %s); trying the next one", name, type(e).__name__, e)
    raise RuntimeError(f"no upstream retrieval config could be loaded: {last}")


def load_pipeline(settings: dict | None = None):
    """The upstream query pipeline. `retrieval.config: auto` prefers v2 (BM25 + re-ranker) and falls
    back to v2-bm25 (no model download) when the re-ranker can't be loaded."""
    settings = settings or load_settings()
    root = ensure_importable(settings)
    return _pipeline(str(settings.get("retrieval", {}).get("config", "auto")), str(root))


__all__ = ["ROOT", "ensure_importable", "load_pipeline", "upstream_commit", "upstream_root"]
