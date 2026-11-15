import numpy as np
import pytest

from latent_diffusion.gen_eval import (prdc, membership_attack, nearest_distance, privacy_distances,
                                            validate_ecgs, feature_distribution, ECGEncoder)


def test_prdc_copy_collapse_and_outlier_controls():
    rng = np.random.default_rng(9)
    real = rng.normal(size=(120, 5))
    same = prdc(real, real)
    assert same['precision'] == same['recall'] == same['coverage'] == 1
    collapsed = prdc(real, np.repeat(real[:1], len(real), axis=0))
    assert collapsed['precision'] == 1 and collapsed['coverage'] < .2
    outlier = prdc(real, real+100)
    assert outlier['precision'] == outlier['coverage'] == 0


def test_privacy_controls_detect_copying():
    rng = np.random.default_rng(22)
    members = rng.normal(size=(100, 8))
    nonmembers = rng.normal(size=(100, 8))
    copied = privacy_distances(members, nonmembers, members)
    assert copied['exact_copy_fraction'] == 1
    assert copied['ratio_of_medians'] < 1e-6
    attack = membership_attack(members, nonmembers, members)
    assert attack['auc'] == 1
    independent = membership_attack(members, nonmembers, rng.normal(size=(100, 8)))
    assert .3 < independent['auc'] < .7


def test_distances_and_input_rejection():
    a = np.array([[0., 0.], [3., 4.]])
    assert np.allclose(nearest_distance(a, a[:1]), [0, 5/np.sqrt(2)])
    for x in (np.zeros((0, 1000, 12)), np.full((1, 1000, 12), np.nan), np.zeros((1, 12, 1000))):
        with pytest.raises(ValueError):
            validate_ecgs(x)
    with pytest.raises(ValueError):
        prdc(np.zeros((3, 8)), np.zeros((10, 8)))


def test_missing_features_are_reported_without_imputation():
    a = np.zeros((10, 25)); b = np.ones((10, 25)); b[:, 0] = np.nan
    results = feature_distribution(a, b)
    assert results['hr']['missing_synthetic_fraction'] == 1
    assert 'wasserstein' not in results['hr']
    assert results['sdnn_ms']['wasserstein'] == 1
    assert results['sdnn_ms']['wasserstein_over_real_iqr'] is None


def test_encoder_is_frozen_and_has_finite_embeddings():
    e = ECGEncoder()
    assert not any(p.requires_grad for p in e.model.parameters())
    a = e(np.random.default_rng(0).normal(size=(2, 1000, 12)).astype(np.float32))
    assert a.shape == (2, 64) and np.isfinite(a).all()
