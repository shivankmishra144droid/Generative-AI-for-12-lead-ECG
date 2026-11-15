"""Study 9: SSSD in latent space, cosine v diffusion, joint CFG and DDIM.

No modifications to the original SSSD implementation. Conditioning bypasses its
class-only embedding and enters the existing residual group's label projection.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def cosine_alpha_bar(steps=1000, offset=.008):
    if steps < 2:
        raise ValueError('At least two diffusion steps are required')
    t = torch.linspace(0, steps, steps+1, dtype=torch.float64) / steps
    curve = torch.cos((t+offset)/(1+offset)*math.pi/2).square()
    curve /= curve[0].clone()
    beta = (1 - curve[1:]/curve[:-1]).clamp(1e-8, .999)
    return torch.cumprod(1-beta, 0).float()


class ConditionEncoder(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.class_embed = nn.Embedding(4, dim)
        self.numeric = nn.Sequential(nn.Linear(6, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.null = nn.Parameter(torch.zeros(dim))

    def forward(self, labels, values, drop=None):
        c = self.class_embed(labels) + self.numeric(values)
        if drop is not None:
            c = torch.where(drop[:, None], self.null[None], c)
        return c


class LatentSSSD(nn.Module):
    def __init__(self, channels=16, width=32, layers=4, state=16, condition_dim=64):
        super().__init__()
        from synthetic_generation.model import SSSD_ECG
        self.config = dict(channels=channels, width=width, layers=layers, state=state,
                           condition_dim=condition_dim)
        self.conditions = ConditionEncoder(condition_dim)
        self.net = SSSD_ECG(
            in_channels=channels, out_channels=channels, res_channels=width, skip_channels=width,
            num_res_layers=layers, diffusion_step_embed_dim_in=64,
            diffusion_step_embed_dim_mid=128, diffusion_step_embed_dim_out=128,
            s4_lmax=125, s4_d_state=state, s4_dropout=0., s4_bidirectional=True,
            s4_layernorm=True, label_embed_classes=4, label_embed_dim=condition_dim)
        # The class-only lookup is superseded by ConditionEncoder; do not optimize unused weights.
        self.net.embedding.requires_grad_(False)
        # Some frozen S4 buffers are expanded views. CUDA transfers materialize
        # them, but CPU checkpoint loading cannot copy into overlapping storage.
        for module in self.net.modules():
            for name, value in module.named_buffers(recurse=False):
                if value is not None and not value.is_contiguous():
                    module._buffers[name] = value.contiguous()

    def forward(self, z, t, labels, values, drop=None):
        c = self.conditions(labels, values, drop)
        h = self.net.init_conv(z)
        return self.net.final_conv(self.net.residual_layer((h, c, t.reshape(-1, 1))))


class LatentDiffusion(nn.Module):
    def __init__(self, denoiser, steps=1000, condition_dropout=.15):
        super().__init__()
        if not 0 < condition_dropout < 1:
            raise ValueError('CFG needs nonzero conditional and unconditional training')
        self.denoiser = denoiser
        self.steps = steps
        self.condition_dropout = condition_dropout
        self.register_buffer('alpha_bar', cosine_alpha_bar(steps))

    def noise_target(self, z, noise, t):
        a = self.alpha_bar[t].view(-1, 1, 1).sqrt()
        s = (1-self.alpha_bar[t]).view(-1, 1, 1).sqrt()
        return a*z+s*noise, a*noise-s*z

    def loss(self, z, labels, values):
        t = torch.randint(self.steps, (len(z),), device=z.device)
        noisy, target = self.noise_target(z, torch.randn_like(z), t)
        drop = torch.rand(len(z), device=z.device) < self.condition_dropout
        return F.mse_loss(self.denoiser(noisy, t, labels, values, drop), target)

    @torch.no_grad()
    def sample(self, labels, values, sampling_steps=50, guidance=2., seed=9009):
        if not 1 <= sampling_steps <= self.steps or not math.isfinite(guidance) or guidance < 0:
            raise ValueError('Invalid DDIM steps or CFG scale')
        device = self.alpha_bar.device
        generator = torch.Generator(device=device).manual_seed(seed)
        z = torch.randn((len(labels), self.denoiser.config['channels'], 125), device=device,
                        generator=generator)
        timeline = torch.linspace(self.steps-1, 0, sampling_steps, device=device).round().long()
        self.eval()
        for i, tick in enumerate(timeline):
            t = tick.expand(len(z))
            vc = self.denoiser(z, t, labels, values)
            if guidance != 1:
                vu = self.denoiser(z, t, labels, values, torch.ones_like(labels, dtype=torch.bool))
                v = vu + guidance*(vc-vu)
            else:
                v = vc
            a = self.alpha_bar[tick].sqrt()
            s = (1-self.alpha_bar[tick]).sqrt()
            x0, eps = a*z-s*v, s*z+a*v
            previous = self.alpha_bar[timeline[i+1]] if i+1 < len(timeline) else z.new_tensor(1.)
            z = previous.sqrt()*x0 + (1-previous).sqrt()*eps  # DDIM eta=0
        return z
