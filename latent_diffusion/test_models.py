import io
import numpy as np
import pytest
import torch

from latent_diffusion.data import condition_values
from latent_diffusion.latent.vae import ECGVAE, vae_loss
from latent_diffusion.latent.latent_diffusion import ConditionEncoder, LatentDiffusion, LatentSSSD, cosine_alpha_bar

torch.set_num_threads(2)


def test_compression_backward_and_checkpoint():
    torch.manual_seed(9)
    model = ECGVAE(width=8)
    x = torch.randn(2, 12, 1000)
    r, mu, lv = model(x)
    assert r.shape == x.shape and mu.shape == (2, 16, 125)
    loss, terms = vae_loss(r, x, mu, lv)
    loss.backward()
    assert all(torch.isfinite(v) for v in terms.values())
    assert model.encoder[0].weight.grad.abs().sum() > 0
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf); buf.seek(0)
    clone = ECGVAE(width=8); clone.load_state_dict(torch.load(buf, weights_only=True))
    assert torch.equal(model(x, sample=False)[0], clone(x, sample=False)[0])


def test_invalid_shapes_and_missing_conditions():
    with pytest.raises(ValueError):
        ECGVAE().encode(torch.zeros(2, 1000, 12))
    v = condition_values([45, np.nan], [300, 55], [np.nan, 1])
    assert np.isfinite(v).all()
    assert np.array_equal(v[:, 3:], [[1, 0, 0], [0, 1, 1]])
    assert v[0, 0] == -.75


def test_cosine_terminal_and_v_inverse():
    ab = cosine_alpha_bar()
    assert (ab[:-1] > ab[1:]).all() and ab[-1] < 1e-7
    a, s = ab[[0, 499, 999]].sqrt()[:, None, None], (1-ab[[0, 499, 999]]).sqrt()[:, None, None]
    x, noise = torch.randn(3, 16, 125), torch.randn(3, 16, 125)
    xt, v = a*x+s*noise, a*noise-s*x
    assert torch.allclose(a*xt-s*v, x, atol=1e-6)
    assert torch.allclose(s*xt+a*v, noise, atol=1e-6)


def test_joint_null_condition_has_no_label_or_numeric_leak():
    e = ConditionEncoder(8)
    drop = torch.ones(2, dtype=torch.bool)
    a = e(torch.tensor([0, 1]), torch.randn(2, 6), drop)
    b = e(torch.tensor([3, 2]), torch.randn(2, 6)*10, drop)
    assert torch.equal(a, b)


def test_sssd_forward_backward_and_ddim_reproducibility():
    net = LatentSSSD(width=8, layers=1, state=4, condition_dim=8)
    diffusion = LatentDiffusion(net)
    y, values = torch.tensor([0, 3]), torch.zeros(2, 6)
    loss = diffusion.loss(torch.randn(2, 16, 125), y, values)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())
    a = diffusion.sample(y, values, sampling_steps=3, seed=123)
    b = diffusion.sample(y, values, sampling_steps=3, seed=123)
    assert a.shape == (2, 16, 125) and torch.isfinite(a).all() and torch.equal(a, b)
    with pytest.raises(ValueError):
        diffusion.sample(y, values, sampling_steps=1001)


def test_sssd_checkpoint_loads_on_cpu_without_overlapping_buffers():
    config = dict(width=8, layers=1, state=4, condition_dim=8)
    original = LatentDiffusion(LatentSSSD(**config))
    buffer = io.BytesIO()
    torch.save(original.state_dict(), buffer)
    buffer.seek(0)
    restored = LatentDiffusion(LatentSSSD(**config))
    restored.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=True), strict=True)
    for key, value in original.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key
