"""Study 4, D5 escalation: a learned model of the full beat residual (report/WORKFLOW_4.md, Phase 2 ladder rung R2).

Target: r = highpass(x8) - place(template, R), i.e. everything the current generator models with the stationary GP
(beat-to-beat variation + leftover), on the 8 independent leads, 1000 samples.
Condition: c = place(template, R) (8 channels, concatenated with x_t) + the class one-hot. This is exactly what exists
when generating: a (generated) template placed at sampled R-peaks.

Phase 2-3 escalation found that the giveaway is the residual's time-domain structure (heavy tails, edges, beat
locking, waveform-like per-beat deviations), which a stationary or envelope-modulated GP cannot carry; a model of the
whole residual given its template also avoids the pairing confound of swapping only the deviations.

Backbone: the frozen SSSD_ECG class (synthetic_generation/model.py, not edited), built with in_channels = 16.
Diffusion: T = 200, beta 5e-4 -> 0.1 (sqrt(alpha_bar_T) = 0.006; the 1e-4 -> 0.02 schedule leaves 36 % of the signal at
t = T, a train/sample mismatch REPORT 5.2 already found), v-prediction, ancestral DDPM sampling.
Loss and sampler are new here because the frozen ones take no conditioning channels."""
from __future__ import annotations

import torch

from synthetic_generation.model import SSSD_ECG
from synthetic_generation.utils import calc_diffusion_hyperparams, hyperparams_to_device

DIFFUSION = {"T": 200, "beta_0": 5e-4, "beta_T": 0.1}          # the project schedule (_run_vpred.py, _study3_build.py)
NET = {"in_channels": 16, "out_channels": 8, "num_res_layers": 8, "res_channels": 64, "skip_channels": 64,
       "diffusion_step_embed_dim_in": 128, "diffusion_step_embed_dim_mid": 512, "diffusion_step_embed_dim_out": 512,
       "s4_lmax": 1000, "s4_d_state": 64, "s4_dropout": 0.0, "s4_bidirectional": 1, "s4_layernorm": 1,
       "label_embed_classes": 4, "label_embed_dim": 128}


def build(device) -> tuple[SSSD_ECG, dict]:
    return SSSD_ECG(**NET).to(device), hyperparams_to_device(calc_diffusion_hyperparams(**DIFFUSION), device)


def v_loss(net, x0: torch.Tensor, cond: torch.Tensor, label: torch.Tensor, dh: dict) -> torch.Tensor:
    """x0 (B, 8, L) scaled residual, cond (B, 8, L) scaled template signal, label (B, 4) one-hot."""
    b = x0.shape[0]
    ab = dh["Alpha_bar"]
    t = torch.randint(dh["T"], size=(b, 1, 1), device=x0.device)
    z = torch.randn_like(x0)
    xt = ab[t].sqrt() * x0 + (1 - ab[t]).sqrt() * z
    v = ab[t].sqrt() * z - (1 - ab[t]).sqrt() * x0
    out = net((torch.cat([xt, cond], 1), label, t.view(b, 1).float()))
    return torch.mean((out - v) ** 2)


@torch.no_grad()
def sample(net, cond: torch.Tensor, label: torch.Tensor, dh: dict, generator: torch.Generator | None = None) -> torch.Tensor:
    """Ancestral DDPM with v-prediction converted to eps. cond (B, 8, L), label (B, 4)."""
    T, alpha, ab, sigma = dh["T"], dh["Alpha"], dh["Alpha_bar"], dh["Sigma"]
    shape = (cond.shape[0], 8, cond.shape[2])
    x = torch.randn(shape, device=cond.device, generator=generator)
    for t in range(T - 1, -1, -1):
        steps = torch.full((shape[0], 1), float(t), device=cond.device)
        v = net((torch.cat([x, cond], 1), label, steps))
        eps = (1 - ab[t]).sqrt() * x + ab[t].sqrt() * v
        x = (x - (1 - alpha[t]) / (1 - ab[t]).sqrt() * eps) / alpha[t].sqrt()
        if t > 0:
            x = x + sigma[t] * torch.randn(shape, device=cond.device, generator=generator)
    return x
