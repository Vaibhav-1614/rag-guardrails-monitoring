"""Query/passage embedders.

- ``bge``: BAAI/bge-small-en-v1.5 via fastembed (ONNX). Chunk vectors come from the upstream
  index artifacts, so only queries are encoded at runtime.
- ``lsa``: TF-IDF + truncated SVD fitted on the corpus. Zero downloads, no PyTorch; used
  when the model can't be loaded (offline CI, slim image) and cached under models/.

Both return L2-normalised float32 rows, so cosine similarity is a dot product.
"""
from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import Protocol

import numpy as np

from rag.config import load_settings, resolve
from rag.corpus import TICKERS, Corpus, load_corpus
from rag.upstream import upstream_root

logger = logging.getLogger(__name__)


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x / (np.linalg.norm(x) + 1e-9)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...

    @property
    def chunk_matrix(self) -> np.ndarray: ...


class LSAEmbedder:
    name = "lsa"

    def __init__(self, corpus: Corpus, dims: int = 256, cache_dir: str = "models") -> None:
        import joblib
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [c.text for c in corpus.chunks]
        digest = hashlib.sha1(("|".join(c.chunk_id for c in corpus.chunks) + f"|{dims}|v2").encode()).hexdigest()[:12]
        cache = resolve(cache_dir) / f"lsa_{digest}.joblib"
        if cache.exists():
            try:
                self._vec, self._svd, self._matrix = joblib.load(cache)
                self.dim = self._matrix.shape[1]
                return
            except Exception:  # corrupt/old cache: refit
                logger.warning("Ignoring unreadable LSA cache %s", cache)
        self._vec = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2, max_features=60000, dtype=np.float32)
        X = self._vec.fit_transform(texts)
        self._svd = TruncatedSVD(n_components=min(dims, X.shape[1] - 1), random_state=0)
        self._matrix = _normalize(self._svd.fit_transform(X))
        self.dim = self._matrix.shape[1]
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump((self._vec, self._svd, self._matrix), cache)
        except OSError:
            pass

    def encode(self, texts: list[str]) -> np.ndarray:
        return _normalize(self._svd.transform(self._vec.transform(texts)))

    def novelty_features(self, texts: list[str]) -> np.ndarray:
        """Share of each text's content words the corpus has never seen, as an (n, 1) column.

        LSA maps unseen words ("Netflix", "Bitcoin") to nothing, so a shift toward topics outside the
        filings is invisible in the embedding itself. Drift tests append this column."""
        vocab, analyze = self._vec.vocabulary_, self._vec.build_analyzer()
        out = []
        for t in texts:
            toks = [w for w in analyze(t) if len(w) > 2 and not w.isdigit()]
            out.append(sum(w not in vocab for w in toks) / len(toks) if toks else 0.0)
        return np.asarray(out, dtype=np.float32)[:, None]

    @property
    def chunk_matrix(self) -> np.ndarray:
        return self._matrix


class BGEEmbedder:
    """bge-small through fastembed (ONNX, no PyTorch), the same model and chunk vectors the upstream
    service uses for its hybrid configs."""

    name = "bge"

    def __init__(self, corpus: Corpus, model_name: str, settings: dict) -> None:
        from fastembed import TextEmbedding  # optional dependency

        emb_dir = upstream_root(settings) / "data" / "index" / "v2" / "embeddings"
        matrix = np.vstack([np.load(emb_dir / f"{t}_bge.npy").astype(np.float32) for t in TICKERS])
        if matrix.shape[0] != len(corpus):
            raise ValueError(f"bge artifacts have {matrix.shape[0]} rows, corpus has {len(corpus)} chunks")
        self._model = TextEmbedding(model_name)
        self._matrix = _normalize(matrix)
        self.dim = self._matrix.shape[1]

    def encode(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.asarray(list(self._model.query_embed(texts)), dtype=np.float32))

    @property
    def chunk_matrix(self) -> np.ndarray:
        return self._matrix


@lru_cache(maxsize=4)
def _build(backend: str, model: str, dims: int, cache_dir: str, settings_key: str) -> Embedder:
    settings = load_settings()
    corpus = load_corpus(settings)
    if backend in ("auto", "bge"):
        try:
            return BGEEmbedder(corpus, model, settings)
        except Exception as e:  # ImportError, no network for weights, ...
            if backend == "bge":
                raise
            logger.warning("bge embedder unavailable (%s: %s); falling back to LSA", type(e).__name__, e)
    return LSAEmbedder(corpus, dims=dims, cache_dir=cache_dir)


def get_embedder(settings: dict | None = None) -> Embedder:
    settings = settings or load_settings()
    e = settings.get("embedder", {})
    return _build(
        str(e.get("backend", "auto")),
        str(e.get("bge_model", "BAAI/bge-small-en-v1.5")),
        int(e.get("lsa_dims", 256)),
        str(e.get("cache_dir", "models")),
        str(upstream_root(settings)),
    )
