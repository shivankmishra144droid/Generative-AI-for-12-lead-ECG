"""Unit checks for synthetic_generation/realism/features.py (PLAN_features.md, phase F0)."""
import numpy as np

from synthetic_generation.realism import features as F


def pulse_train(hr=75.0, width=2.0, T=1000, noise=0.02, seed=0):
    """8-lead signal of R-S 'QRS' complexes (Gaussian lobes of sd `width` samples) at a fixed heart rate, plus a small T wave."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    rr = 100 * 60 / hr
    x = np.zeros(T)
    for c in np.arange(rr / 2, T, rr):
        x += (3 * np.exp(-0.5 * ((t - c) / width) ** 2) - 1.2 * np.exp(-0.5 * ((t - c - 2 * width) / width) ** 2)
              + 0.5 * np.exp(-0.5 * ((t - c - 28) / 6) ** 2))           # R, S (asymmetric QRS, as in real ECGs), T
    return np.tile(x, (8, 1)) * np.linspace(0.6, 1.2, 8)[:, None] + noise * rng.standard_normal((8, T))


def feat(x8):
    return dict(zip(F.NAMES, F.features_x8(x8)))


def test_heart_rate_recovered():
    for hr in (60.0, 75.0, 110.0):
        assert abs(feat(pulse_train(hr))["hr"] - hr) < 2.0


def test_regular_rhythm_has_low_variability():
    f = feat(pulse_train(75.0))
    assert f["sdnn_ms"] < 15 and f["rr_cv"] < 0.02


def test_qrs_width_grows_with_pulse_width():
    assert feat(pulse_train(width=4.0))["qrs_width_ms"] > feat(pulse_train(width=1.5))["qrs_width_ms"]


def test_gaussian_kurtosis_and_edge_ratio():
    z = np.random.default_rng(1).standard_normal((8, 20000))
    assert abs(np.median(F.kurtosis(z)) - 3.0) < 0.15
    assert abs(F.edge_ratio((z[:, :1000] ** 2).mean(0)) - 1.0) < 0.2


def test_all_features_finite_on_clean_signal():
    v = F.features_x8(pulse_train())
    assert np.isfinite(v).all(), dict(zip(F.NAMES, v))
