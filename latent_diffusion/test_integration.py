import json
import numpy as np
import pytest

from latent_diffusion.common import DATA, HOME, audit, load_split, write_json
from latent_diffusion.published import convert_arrays


def test_outputs_and_test_split_are_guarded(tmp_path):
    with pytest.raises(ValueError):
        load_split('test')
    with pytest.raises(ValueError):
        write_json(tmp_path / 'outside.json', {'a': 1})


def test_published_millivolt_lead_conversion_and_exact_one_rhythm():
    from synthetic_generation.leads import reconstruct_12
    stats = json.loads((DATA / 'normalization_stats.json').read_text())
    rng = np.random.default_rng(42)
    z = rng.normal(size=(5, 8, 1000)).astype(np.float32)
    expected = reconstruct_12(z)
    indices = [0, 1, 6, 7, 8, 9, 10, 11]
    mv = z*np.asarray(stats['std'])[indices][None, :, None]+np.asarray(stats['mean'])[indices][None, :, None]
    # Upstream independent basis I, aVF, V1..V6 rather than I, II.
    mv[:, 1] -= mv[:, 0]/2
    labels = np.r_[np.eye(4, dtype=int), np.array([[1, 0, 0, 1]])]
    actual, y, keep = convert_arrays(mv, labels, ['SR', 'AFIB', 'STACH', 'SBRAD'],
                                     ['I','aVF','V1','V2','V3','V4','V5','V6'], stats)
    assert np.array_equal(y, [0, 1, 2, 3]) and np.array_equal(keep, [0, 1, 2, 3])
    assert np.allclose(actual, expected[:4], atol=2e-6)
    with pytest.raises(ValueError):
        convert_arrays(mv, labels[:, :3], ['SR','AFIB','STACH'], ['I','aVF','V1','V2','V3','V4','V5','V6'], stats)


def test_patient_split_is_disjoint_and_condition_rows_align():
    path = HOME / 'artifacts/partition.npz'
    if not path.exists():
        pytest.skip('Run data preparation for the local-data integration check')
    p = np.load(path)
    m = np.load(HOME / 'artifacts/train_conditions.npz')
    assert not set(m['patient'][p['fit']]) & set(m['patient'][p['dev']])
    assert len(np.union1d(p['fit'], p['dev'])) == len(m['patient'])
    assert m['values'].shape == (len(m['patient']), 6)
    assert np.isfinite(m['values']).all()


def test_ecgsyn_rr_endpoint_regression_without_package_mutation():
    from latent_diffusion.generate import optional_dependencies
    optional_dependencies()
    pytest.importorskip('neurokit2')
    import importlib
    from latent_diffusion.ecgsyn_adapter import simulate, positive_rr_resample
    module = importlib.import_module('neurokit2.ecg.ecg_simulate')
    original = module.signal_resample
    rr = positive_rr_resample(np.ones(16)*.57, 1, 100)
    assert len(rr) == 1600 and np.min(rr) > .5
    # This exact seed/HR hung in the upstream zero-tail RR loop.
    x = simulate(duration=10, sampling_rate=100, heart_rate=105.1546401977539,
                 heart_rate_std=1., noise=.01, method='multileads', random_state=9161)
    assert x.shape == (1000, 12) and np.isfinite(x.to_numpy()).all()
    assert module.signal_resample is original
