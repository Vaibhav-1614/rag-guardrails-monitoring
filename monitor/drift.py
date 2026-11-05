"""Drift statistics (design doc §8.1). Pure numpy/scipy/sklearn; no I/O.

- psi: Population Stability Index of a score distribution vs. reference (alert > 0.2).
- mmd_test: RBF-kernel maximum mean discrepancy with a permutation p-value.
- classifier_two_sample_test: can a classifier tell reference from current embeddings? (AUC, p-value)
- centroid_distance: cosine distance between mean embeddings.
- rolling_band: is today's value outside mean +/- k*sigma of the trailing window?
- rate_spike: binomial z-test of today's trigger rate vs. the trailing rate.
- emerging_clusters: clusters of current queries that are far from everything in the reference.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """PSI with bin edges at reference quantiles (robust to skewed scores)."""
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    if len(reference) < 2 or len(current) < 2:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(min(reference.min(), current.min()), max(reference.max(), current.max()) + 1e-9, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    c = np.histogram(current, edges)[0] / len(current)
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    xx = (X * X).sum(1)[:, None]
    yy = (Y * Y).sum(1)[None, :]
    return np.maximum(xx + yy - 2 * X @ Y.T, 0.0)


def mmd_test(X: np.ndarray, Y: np.ndarray, n_perm: int = 200, seed: int = 0, max_n: int = 600) -> dict:
    """Unbiased MMD^2 with an RBF kernel (median-heuristic bandwidth) and a permutation p-value."""
    rng = np.random.default_rng(seed)
    if len(X) > max_n:
        X = X[rng.choice(len(X), max_n, replace=False)]
    if len(Y) > max_n:
        Y = Y[rng.choice(len(Y), max_n, replace=False)]
    Z = np.vstack([X, Y])
    D = _sq_dists(Z, Z)
    med = np.median(D[np.triu_indices_from(D, 1)])
    K = np.exp(-D / (med if med > 0 else 1.0))
    n, m = len(X), len(Y)

    def stat(idx: np.ndarray) -> float:
        a, b = idx[:n], idx[n:]
        kxx = K[np.ix_(a, a)]
        kyy = K[np.ix_(b, b)]
        kxy = K[np.ix_(a, b)]
        return ((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())

    base = np.arange(n + m)
    observed = stat(base)
    null = np.array([stat(rng.permutation(base)) for _ in range(n_perm)])
    p = (1 + np.sum(null >= observed)) / (n_perm + 1)
    return {"mmd2": float(observed), "p_value": float(p), "n_ref": n, "n_cur": m}


def classifier_two_sample_test(X: np.ndarray, Y: np.ndarray, folds: int = 5, seed: int = 0) -> dict:
    """Out-of-fold AUC of a logistic regression separating reference (X) from current (Y).

    Under H0 the out-of-fold AUC is a Mann-Whitney statistic centred at 0.5, so the p-value uses
    the normal approximation var = (n1 + n2 + 1) / (12 n1 n2).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    Z = np.vstack([X, Y])
    y = np.r_[np.zeros(len(X)), np.ones(len(Y))]
    oof = np.zeros(len(Z))
    skf = StratifiedKFold(n_splits=min(folds, int(min(len(X), len(Y)))), shuffle=True, random_state=seed)
    for tr, te in skf.split(Z, y):
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
        clf.fit(Z[tr], y[tr])
        oof[te] = clf.predict_proba(Z[te])[:, 1]
    auc = float(roc_auc_score(y, oof))
    n1, n2 = len(X), len(Y)
    sd = math.sqrt((n1 + n2 + 1) / (12 * n1 * n2))
    p = float(1 - stats.norm.cdf((auc - 0.5) / sd))
    return {"auc": round(auc, 4), "p_value": p, "n_ref": n1, "n_cur": n2}


def centroid_distance(X: np.ndarray, Y: np.ndarray) -> float:
    a, b = X.mean(0), Y.mean(0)
    return float(1 - a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@dataclass
class BandResult:
    value: float
    mean: float
    std: float
    lower: float
    upper: float
    outside: bool
    n_history: int

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def rolling_band(history: list[float], today: float, k: float = 3.0, min_std: float = 0.0,
                 min_abs_change: float = 0.0, min_history: int = 5, robust: bool = True) -> BandResult:
    """Today vs. centre +/- k*scale of the trailing window. Needs ``min_history`` points and a change of
    at least ``min_abs_change`` (so tiny absolute moves on a flat series don't page anyone).

    robust (default): centre = median, scale = 1.4826 * MAD, so a few already-drifted days inside the
    trailing window can't widen the band and silently absorb a persistent shift.
    """
    h = np.asarray([x for x in history if x is not None and not math.isnan(x)], float)
    if len(h) < min_history or today is None or math.isnan(today):
        return BandResult(today if today is not None else float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), False, len(h))
    if robust:
        mean = float(np.median(h))
        std = float(max(1.4826 * np.median(np.abs(h - mean)), min_std))
    else:
        mean, std = float(h.mean()), float(max(h.std(ddof=1) if len(h) > 1 else 0.0, min_std))
    lo, hi = mean - k * std, mean + k * std
    outside = (today < lo or today > hi) and abs(today - mean) >= min_abs_change
    return BandResult(float(today), mean, std, lo, hi, bool(outside), len(h))


def rate_spike(count_today: int, n_today: int, count_base: int, n_base: int, z_min: float = 4.0,
               ratio_min: float = 2.0, min_count: int = 5) -> dict:
    """Binomial z-test: is today's rate far above the trailing baseline rate?"""
    if n_today == 0:
        return {"spike": False, "rate_today": None, "rate_base": None, "z": None}
    p_today = count_today / n_today
    p_base = max(count_base / n_base if n_base else 0.0, 0.5 / max(n_base, 1))
    z = (p_today - p_base) / math.sqrt(p_base * (1 - p_base) / n_today) if 0 < p_base < 1 else 0.0
    spike = z >= z_min and p_today >= ratio_min * p_base and count_today >= min_count
    return {"spike": bool(spike), "rate_today": round(p_today, 4), "rate_base": round(p_base, 4), "z": round(z, 2),
            "count_today": count_today, "n_today": n_today}


@dataclass
class Cluster:
    size: int
    share: float
    novelty: float
    top_terms: list[str]
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"size": self.size, "share": round(self.share, 3), "novelty": round(self.novelty, 3),
                "top_terms": self.top_terms, "examples": self.examples}


def _nn_distance(A: np.ndarray, B: np.ndarray, exclude_self: bool = False) -> np.ndarray:
    sims = A @ B.T
    if exclude_self:
        np.fill_diagonal(sims, -np.inf)
    return 1 - sims.max(1)


def emerging_clusters(ref: np.ndarray, cur: np.ndarray, cur_texts: list[str], quantile: float = 0.95,
                      min_size: int = 8, min_share: float = 0.05, max_clusters: int = 6, seed: int = 0) -> dict:
    """Queries farther from the reference than ``quantile`` of the reference's own nearest-neighbour
    distances are 'novel'; novel queries are clustered, and big enough clusters are emerging topics."""
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    if len(cur) < min_size or len(ref) < 10:
        return {"novel_share": 0.0, "threshold": None, "clusters": []}
    ref_nn = _nn_distance(ref, ref, exclude_self=True)
    tau = float(np.quantile(ref_nn, quantile))
    d = _nn_distance(cur, ref)
    novel = np.where(d > tau)[0]
    out = {"novel_share": round(len(novel) / len(cur), 4), "threshold": round(tau, 4), "clusters": []}
    if len(novel) < min_size:
        return out
    k = max(1, min(max_clusters, len(novel) // min_size))
    labels = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(cur[novel])
    texts = [cur_texts[i] for i in novel]
    try:
        vec = TfidfVectorizer(stop_words="english", min_df=1)
        T = vec.fit_transform(texts)
        vocab = np.array(vec.get_feature_names_out())
    except ValueError:
        T, vocab = None, np.array([])
    clusters = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) < min_size or len(idx) / len(cur) < min_share:
            continue
        terms = []
        if T is not None and len(vocab):
            w = np.asarray(T[idx].mean(0)).ravel()
            terms = vocab[np.argsort(-w)[:6]].tolist()
        clusters.append(Cluster(len(idx), len(idx) / len(cur), float(d[novel][idx].mean()), terms,
                                [texts[i][:120] for i in idx[:3]]))
    out["clusters"] = [c.to_dict() for c in sorted(clusters, key=lambda c: -c.size)]
    return out
