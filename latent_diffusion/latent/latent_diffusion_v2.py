"""V2 latent diffusion: any latent length, optional R-peak-map conditioning (Phase D), DDIM with eta.

v1 classes in latent_diffusion.py are untouched. Same cosine schedule, v objective and joint CFG as v1.
R-map input: 2 extra channels at latent resolution, [map, valid flag]. A dropped map is all zeros (flag 0).
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .latent_diffusion import ConditionEncoder, cosine_alpha_bar


class LatentSSSDv2(nn.Module):
    def __init__(self, channels=16, length=125, width=64, layers=8, state=32, condition_dim=64, rhythm=False):
        super().__init__()
        from synthetic_generation.model import SSSD_ECG
        self.config = dict(channels=channels, length=length, width=width, layers=layers, state=state,
                           condition_dim=condition_dim, rhythm=rhythm)
        self.conditions = ConditionEncoder(condition_dim)
        self.net = SSSD_ECG(
            in_channels=channels + (2 if rhythm else 0), out_channels=channels, res_channels=width,
            skip_channels=width, num_res_layers=layers, diffusion_step_embed_dim_in=64,
            diffusion_step_embed_dim_mid=128, diffusion_step_embed_dim_out=128,
            s4_lmax=length, s4_d_state=state, s4_dropout=0., s4_bidirectional=True,
            s4_layernorm=True, label_embed_classes=4, label_embed_dim=condition_dim)
        self.net.embedding.requires_grad_(False)
        for module in self.net.modules():           # same CPU-loading fix as v1 LatentSSSD
            for name, value in module.named_buffers(recurse=False):
                if value is not None and not value.is_contiguous():
                    module._buffers[name] = value.contiguous()

    def forward(self, z, t, labels, values, drop=None, rmap=None):
        c = self.conditions(labels, values, drop)
        if self.config['rhythm']:
            if rmap is None:
                rmap = z.new_zeros(len(z), 2, z.shape[-1])
            z = torch.cat([z, rmap], 1)
        h = self.net.init_conv(z)
        return self.net.final_conv(self.net.residual_layer((h, c, t.reshape(-1, 1))))


class LatentDiffusionV2(nn.Module):
    def __init__(self, denoiser, steps=1000, p_vec=.1, p_map=.1, p_both=.05):
        super().__init__()
        self.denoiser = denoiser
        self.steps = steps
        self.p = (p_vec, p_map, p_both)
        self.register_buffer('alpha_bar', cosine_alpha_bar(steps))

    def noise_target(self, z, noise, t):
        a = self.alpha_bar[t].view(-1, 1, 1).sqrt()
        s = (1 - self.alpha_bar[t]).view(-1, 1, 1).sqrt()
        return a*z + s*noise, a*noise - s*z

    def drops(self, n, device, generator=None):
        u = torch.rand(n, 3, device=device, generator=generator)
        both = u[:, 2] < self.p[2]
        return (u[:, 0] < self.p[0]) | both, (u[:, 1] < self.p[1]) | both

    def loss(self, z, labels, values, rmap=None, t=None, noise=None, generator=None, per_sample=False):
        if t is None:
            t = torch.randint(self.steps, (len(z),), device=z.device, generator=generator)
        if noise is None:
            noise = torch.randn(z.shape, device=z.device, generator=generator)
        noisy, target = self.noise_target(z, noise, t)
        dv, dm = self.drops(len(z), z.device, generator)
        if rmap is not None:
            rmap = rmap * (~dm).view(-1, 1, 1)
        err = (self.denoiser(noisy, t, labels, values, dv, rmap) - target).square().mean((1, 2))
        return err if per_sample else err.mean()

    @torch.no_grad()
    def sample(self, labels, values, rmap=None, sampling_steps=50, guidance=2., seed=9009, eta=0.):
        device = self.alpha_bar.device
        g = torch.Generator(device=device).manual_seed(seed)
        cfg = self.denoiser.config
        z = torch.randn((len(labels), cfg['channels'], cfg['length']), device=device, generator=g)
        timeline = torch.linspace(self.steps - 1, 0, sampling_steps, device=device).round().long()
        null = torch.ones(len(labels), dtype=torch.bool, device=device)
        for i, tick in enumerate(timeline):
            t = tick.expand(len(z))
            vc = self.denoiser(z, t, labels, values, None, rmap)
            if guidance != 1:
                vu = self.denoiser(z, t, labels, values, null, None if rmap is None else torch.zeros_like(rmap))
                v = vu + guidance*(vc - vu)
            else:
                v = vc
            a, s = self.alpha_bar[tick].sqrt(), (1 - self.alpha_bar[tick]).sqrt()
            x0, eps = a*z - s*v, s*z + a*v
            prev = self.alpha_bar[timeline[i+1]] if i + 1 < len(timeline) else z.new_tensor(1.)
            cur = self.alpha_bar[tick]
            sigma = eta*torch.sqrt((1 - prev)/(1 - cur).clamp_min(1e-12)*(1 - cur/prev)) if eta else z.new_tensor(0.)
            z = prev.sqrt()*x0 + (1 - prev - sigma**2).clamp_min(0).sqrt()*eps
            if eta and i + 1 < len(timeline):
                z = z + sigma*torch.randn(z.shape, device=device, generator=g)
        return z
