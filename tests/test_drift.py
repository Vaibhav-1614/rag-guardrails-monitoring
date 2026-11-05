import numpy as np

from monitor import drift


def _emb(rng, n, d=32, shift=0.0):
    x = rng.normal(size=(n, d))
    x[:, 0] += shift
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_psi_detects_shift_only():
    rng = np.random.default_rng(0)
    ref = rng.normal(0.6, 0.1, 2000)
    assert drift.psi(ref, rng.normal(0.6, 0.1, 500)) < 0.05
    assert drift.psi(ref, rng.normal(0.4, 0.1, 500)) > 0.2


def test_two_sample_tests():
    rng = np.random.default_rng(1)
    ref = _emb(rng, 300)
    same = drift.classifier_two_sample_test(ref, _emb(rng, 200))
    shifted = drift.classifier_two_sample_test(ref, _emb(rng, 200, shift=1.5))
    assert same["auc"] < 0.62 and shifted["auc"] > 0.8 and shifted["p_value"] < 1e-6
    assert drift.mmd_test(ref, _emb(rng, 200), n_perm=100)["p_value"] > 0.01
    assert drift.mmd_test(ref, _emb(rng, 200, shift=1.5), n_perm=100)["p_value"] < 0.02


def test_rolling_band_and_min_change():
    hist = [0.10, 0.11, 0.09, 0.10, 0.12, 0.10, 0.11]
    assert not drift.rolling_band(hist, 0.12, min_std=0.01, min_abs_change=0.05).outside
    assert drift.rolling_band(hist, 0.30, min_std=0.01, min_abs_change=0.05).outside
    assert not drift.rolling_band(hist[:3], 0.9).outside  # not enough history


def test_rate_spike():
    assert drift.rate_spike(20, 60, 7, 420)["spike"]
    assert not drift.rate_spike(2, 60, 7, 420)["spike"]


def test_emerging_clusters_finds_new_topic():
    rng = np.random.default_rng(2)
    ref = _emb(rng, 300)
    new_dir = np.zeros(32)
    new_dir[5] = 1.0
    novel = new_dir + 0.05 * rng.normal(size=(40, 32))
    novel /= np.linalg.norm(novel, axis=1, keepdims=True)
    cur = np.vstack([_emb(rng, 160), novel])
    texts = ["routine question"] * 160 + ["netflix subscriber growth"] * 40
    out = drift.emerging_clusters(ref, cur, texts, min_size=10)
    assert out["clusters"] and "netflix" in out["clusters"][0]["top_terms"]
