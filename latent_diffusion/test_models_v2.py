"""CPU tests for the V2 modules (AE ladder model, rhythm-conditioned diffusion, R-maps, adherence)."""
import numpy as np
import torch
from .latent.vae_v2 import ECGAEv2, ae_loss
from .latent.latent_diffusion_v2 import LatentSSSDv2, LatentDiffusionV2
from .rhythm_v2 import rmap, regular_peaks, plausible, MAXP
from .sample_v2 import adherence
from .train_diffusion_v2 import sampling_weights


def test_ae_v2_shapes_and_v1_equivalent_size():
    from .latent.vae import ECGVAE
    for nd, L in ((3, 125), (2, 250)):
        m = ECGAEv2(n_down=nd)
        r, mu, lv = m(torch.randn(2, 12, 1000), torch.randn(2, 10))
        assert r.shape == (2, 12, 1000) and mu.shape == (2, 16, L)
        assert torch.isfinite(ae_loss(r, torch.randn(2, 12, 1000), mu, lv, stft=.1)[0])
    assert sum(p.numel() for p in ECGAEv2().parameters()) == sum(p.numel() for p in ECGVAE().parameters())


def small(rhythm):
    torch.manual_seed(0)
    return LatentDiffusionV2(LatentSSSDv2(width=16, layers=2, state=8, rhythm=rhythm)).eval()


def test_null_map_equals_absent_map():
    m = small(True)
    z, t = torch.randn(3, 16, 125), torch.tensor([5, 500, 900])
    lb, vb = torch.tensor([0, 1, 2]), torch.zeros(3, 6)
    with torch.no_grad():
        a = m.denoiser(z, t, lb, vb, None, torch.zeros(3, 2, 125))
        b = m.denoiser(z, t, lb, vb, None, None)
    assert torch.allclose(a, b)


def test_ddim_deterministic_and_eta_changes_output():
    m = small(True)
    lb, vb = torch.tensor([0, 3]), torch.zeros(2, 6)
    rm = torch.from_numpy(rmap(np.stack([regular_peaks(60, np.random.default_rng(i)) for i in range(2)]), np.ones(2, bool), 125))
    a = m.sample(lb, vb, rm, 5, 2., seed=1)
    b = m.sample(lb, vb, rm, 5, 2., seed=1)
    c = m.sample(lb, vb, rm, 5, 2., seed=1, eta=.5)
    assert torch.equal(a, b) and not torch.allclose(a, c)


def test_loss_backward_finite():
    m = small(True).train()
    loss = m.loss(torch.randn(4, 16, 125), torch.tensor([0, 1, 2, 3]), torch.zeros(4, 6), torch.rand(4, 2, 125))
    loss.backward()
    assert torch.isfinite(loss)


def test_rmap_and_regular_peaks():
    rng = np.random.default_rng(0)
    p = regular_peaks(60, rng)
    q = p[p >= 0]
    assert abs(6000/np.diff(q).mean() - 60) < 1.5 and len(p) == MAXP
    m = rmap(p[None], np.array([True]), 125)
    assert m.shape == (1, 2, 125) and m[0, 1].min() == 1 and m[0, 0].max() > .2
    assert rmap(p[None], np.array([False]), 125).sum() == 0
    assert plausible(p[None])[0]


def test_adherence_perfect_and_missing():
    req = np.array([[100, 200, 300, -1]])
    assert adherence(req, [np.array([101, 199, 300])])['missed_fraction'] == 0
    r = adherence(req, [np.array([100, 450])])
    assert abs(r['missed_fraction'] - 2/3) < 1e-9 and abs(r['extra_per_beat'] - 1/3) < 1e-9


def test_sampling_weights_class_uniform():
    rng = np.random.default_rng(0)
    y = rng.choice(4, 2000, p=[.85, .07, .04, .04])
    hr = rng.normal(70, 15, 2000)
    hr[:10] = np.nan
    rows = np.arange(2000)
    w = sampling_weights(y, hr, rows)
    assert abs(w.sum() - 1) < 1e-9
    assert np.allclose([w[y == c].sum() for c in range(4)], .25)
