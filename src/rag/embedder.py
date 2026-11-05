"""Query/passage embedders.

- ``bge``: BAAI/bge-small-en-v1.5 via sentence-transformers. Chunk vectors come from the
  vendored float16 artifacts, so only queries are encoded at runtime.
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
from rag.corpus import Corpus, chunk_files_order, load_corpus

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
        digest = hashlib.sha1(("|".join(c.chunk_id for c in corpus.chunks) + f"|{dims}|v1").encode()).hexdigest()[:12]
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

    @property
    def chunk_matrix(self) -> np.ndarray:
        return self._matrix


class BGEEmbedder:
    name = "bge"

    def __init__(self, corpus: Corpus, model_name: str, settings: dict) -> None:
        from sentence_transformers import SentenceTransformer  # optional dependency

        index_dir = resolve(settings.get("corpus", {}).get("index_dir", "data/index"))
        size = corpus.chunk_size
        mats = [np.load(index_dir / "embeddings" / f"{t}_{size}_bge.npy").astype(np.float32) for t in chunk_files_order(settings)]
        matrix = np.vstack(mats)
        if matrix.shape[0] != len(corpus):
            raise ValueError(f"bge artifacts have {matrix.shape[0]} rows, corpus has {len(corpus)} chunks")
        self._model = SentenceTransformer(model_name, device="cpu")
        self._matrix = _normalize(matrix)
        self.dim = self._matrix.shape[1]

    def encode(self, texts: list[str]) -> np.ndarray:
        return _normalize(self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False))

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
        settings.get("corpus", {}).get("index_dir", "data/index"),
    )
